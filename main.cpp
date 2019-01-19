#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/// Global Variables
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;
char *window_name = const_cast<char *>("Display window");

int display_caption(Mat src, char *caption);

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
        if (display_caption(image, const_cast<char *>("Could not open or find the image")) != 0) { return 0; }
        return -1;
    }

    namedWindow(window_name, WINDOW_AUTOSIZE);// Create a window for display.
    if (display_caption(image, const_cast<char *>("image")) != 0) { return 0; }

    imshow(window_name, image);                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window

    //-------------------------------------------------------------------------------
    /* extracting blue channel*/
    if (display_caption(image, const_cast<char *>("blue channel")) != 0) { return 0; }
    Mat bgr[3];   //destination array
    split(image, bgr);//split source

    //Note: OpenCV uses BGR color order
    imwrite("blue.png", bgr[0]); //blue channel
    imshow(window_name, bgr[0]);
    waitKey(0);

    //-------------------------------------------------------------------------------
    /*converting Lenna to greyscale*/
    if (display_caption(image, const_cast<char *>("greyscale")) != 0) { return 0; }
    Mat greyMat;
    cvtColor(image, greyMat, CV_BGR2GRAY);
    imwrite("greyscale.png", greyMat);
    imshow(window_name, greyMat);
    waitKey(0);

    //-------------------------------------------------------------------------------
    // Applying Gaussian blur
    if (display_caption(image, const_cast<char *>("Gaussian blur")) != 0) { return 0; }
    Mat dst;
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
        GaussianBlur(image, dst, Size(i, i), 0, 0);
    }
    imwrite("Gaussian_blur.png", dst);
    imshow(window_name, dst);
    waitKey(0);

    //-------------------------------------------------------------------------------
    // rotate 90 angles
    if (display_caption(image, const_cast<char *>("rotate")) != 0) { return 0; }
    float angle = 90;

    // get rotation matrix for rotating the image around its center in pixel coordinates
    Point2f center(static_cast<float>((image.cols - 1) / 2.0), static_cast<float>((image.rows - 1) / 2.0));
    Mat rot = getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle, center not relevant
    Rect2f bbox = RotatedRect(Point2f(), image.size(), angle).boundingRect2f();
    // adjust transformation matrix
    rot.at<double>(0, 2) += bbox.width / 2.0 - image.cols / 2.0;
    rot.at<double>(1, 2) += bbox.height / 2.0 - image.rows / 2.0;

    warpAffine(image, dst, rot, bbox.size());
    imwrite("rotated_image.png", dst);
    imshow(window_name, dst);
    waitKey(0);

    
    return 0;
}


int display_caption(Mat src, char *caption) {
    Mat d;
    d = Mat::zeros(src.size(), src.type());
    putText(d, caption,
            Point(20, src.rows / 2),
            CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));

    imshow(window_name, d);
    int c = waitKey(DELAY_CAPTION);
    if (c >= 0) { return -1; }
    return 0;
}
