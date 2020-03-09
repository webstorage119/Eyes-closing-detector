/*
    The eyes closing detector uses eye aspect ratio (EAR) to determine whether eyes are closing. EAR can be
    calculated based on face extracted landmarks given by dlib. This detector work quite ok provided that
    you do not wear glasses and close your eyes for 1-2 seconds. 
    This code is based on dlib, with added features for eyes closing detector.
*/

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;
double eye_aspect_ratio(point p1, point p2, point p3, point p4, point p5, point p6) 
{
    double x = sqrt((p2(0) - p6(0))*(p2(0) - p6(0)) + (p2(1) - p6(1))*(p2(1) - p6(1)));
    double y = sqrt((p3(0) - p5(0))*(p3(0) - p5(0)) + (p3(1) - p5(1))*(p3(1) - p5(1)));
    double z = sqrt((p1(0) - p4(0))*(p1(0) - p4(0)) + (p1(1) - p4(1))*(p1(1) - p4(1)));
    return (x + y) / (2 * z);
}
int main()
{
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp))
            {
                break;
            }
            
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(pose_model(cimg, faces[i]));
            
            const double threshold = 0.2; 

            std::vector<rectangle> rects; 
            for (unsigned long i = 0; i < shapes.size(); ++i)
            {
                // Eyes landmark extract
                long x1 = shapes[i].part(36)(0);
                long y1 = shapes[i].part(37)(1);

                long x2 = shapes[i].part(39)(0);
                long y2 = shapes[i].part(40)(1);

                long x3 = shapes[i].part(42)(0);
                long y3 = shapes[i].part(43)(1);

                long x4 = shapes[i].part(45)(0);
                long y4 = shapes[i].part(46)(1); 

                rects.push_back(rectangle(x1,y1,x2,y2));
                rects.push_back(rectangle(x3,y3,x4,y4));

                // EAR
                point p1 = shapes[i].part(36);
                point p2 = shapes[i].part(37);
                point p3 = shapes[i].part(38);
                point p4 = shapes[i].part(39);
                point p5 = shapes[i].part(40);
                point p6 = shapes[i].part(41);
                double ear1 = eye_aspect_ratio(p1, p2, p3, p4, p5, p6);

                               
                p1 = shapes[i].part(42);
                p2 = shapes[i].part(43);
                p3 = shapes[i].part(44);
                p4 = shapes[i].part(45);
                p5 = shapes[i].part(46);
                p6 = shapes[i].part(47);
                double ear2 = eye_aspect_ratio(p1, p2, p3, p4, p5, p6);

                // Average EAR between left and right eyes
                if ((ear1 + ear2 ) / 2 < threshold) cout << "Eyes are closing." << endl;
            }

            win.clear_overlay();
            win.set_image(cimg);
            render_face_detections(shapes);
            win.add_overlay(rects);
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need to put dlib's model shape_predictor_68_face_landmarks.dat in the same directory with your executable" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

