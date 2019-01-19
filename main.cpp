#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    /* reading and displaying image*/

    if (argc != 2) {
        cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if (!image.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Display window", image);                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window

    //-------------------------------------------------------------------------------
    /* extracting blue channel*/
    Mat bgr[3];   //destination array
    split(image, bgr);//split source

    //Note: OpenCV uses BGR color order
    imwrite("blue.png", bgr[0]); //blue channel
    imshow("blue.png", bgr[0]);
    waitKey(0);

    //-------------------------------------------------------------------------------
    /*converting Lenna to greyscale*/
    Mat greyMat;
    cvtColor(image, greyMat, CV_BGR2GRAY);
    imwrite("greyscale.png", greyMat);
    imshow("greyscale.png", greyMat);
    waitKey(0);

    return 0;
}