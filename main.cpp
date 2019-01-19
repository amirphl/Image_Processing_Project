#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/// Global Variables
int DELAY_CAPTION = 1000;
int MAX_KERNEL_LENGTH = 31;
char *window_name = "Display window";
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
Mat image, bgr[3], grey_image, smoothed_image, rotated_image, resized_image, detected_edges, edges;

///functions
void CannyThreshold(int, void *);

void apply_segmentation(Mat);

int display_caption(Mat, char *);

int main(int argc, char **argv) {
    /// reading and displaying image

    if (argc != 2) {
        cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    // Read the file
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    // Check for invalid input
    if (!image.data) {
        if (display_caption(image, "Could not open or find the image") != 0) { return 0; }
        return -1;
    }

    // Create a window for display
    namedWindow(window_name, WINDOW_AUTOSIZE);
    if (display_caption(image, "image") != 0) { return 0; }

    // Show our image inside it.
    imshow(window_name, image);

    // Wait for a keystroke in the window
    waitKey(0);

    //-------------------------------------------------------------------------------
    /// Extracting blue channel
    if (display_caption(image, "blue channel") != 0) { return 0; }

    //split source
    split(image, bgr);

    //Note: OpenCV uses BGR color order
    //blue channel
    imwrite("blue.png", bgr[0]);
    imshow(window_name, bgr[0]);
    waitKey(0);

    //-------------------------------------------------------------------------------
    ///Converting Lenna to greyscale
    if (display_caption(image, "greyscale") != 0) { return 0; }

    cvtColor(image, grey_image, CV_BGR2GRAY);
    imwrite("greyscale.png", grey_image);
    imshow(window_name, grey_image);
    waitKey(0);

    //-------------------------------------------------------------------------------
    /// Applying Gaussian blur
    if (display_caption(image, "Gaussian blur") != 0) { return 0; }

    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
        GaussianBlur(image, smoothed_image, Size(i, i), 0, 0);
    }
    imwrite("Gaussian_blur.png", smoothed_image);
    imshow(window_name, smoothed_image);
    waitKey(0);

    //-------------------------------------------------------------------------------
    /// Rotate 90 angles
    if (display_caption(image, "rotated image") != 0) { return 0; }
    const float angle = 90;

    // get rotation matrix for rotating the image around its center in pixel coordinates
    Point2f center(static_cast<float>((image.cols - 1) / 2.0), static_cast<float>((image.rows - 1) / 2.0));
    rotated_image = getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle, center not relevant
    Rect2f bbox = RotatedRect(Point2f(), image.size(), angle).boundingRect2f();
    // adjust transformation matrix
    rotated_image.at<double>(0, 2) += bbox.width / 2.0 - image.cols / 2.0;
    rotated_image.at<double>(1, 2) += bbox.height / 2.0 - image.rows / 2.0;

    warpAffine(image, smoothed_image, rotated_image, bbox.size());
    imwrite("rotated_image.png", smoothed_image);
    imshow(window_name, smoothed_image);
    waitKey(0);

    //-------------------------------------------------------------------------------
    // Resize
    if (display_caption(image, "resize") != 0) { return 0; }
    const float m_x = 0.5;
    const float m_y = 1;
    resize(image, resized_image, Size(), m_x, m_y);
    imwrite("resized_image.png", resized_image);
    imshow(window_name, resized_image);
    waitKey(0);

    //-------------------------------------------------------------------------------
    // Edge detection
    if (display_caption(image, "detect edges") != 0) { return 0; }
    /// Create a matrix of the same type and size as src (for edges)
    edges.create(image.size(), image.type());

    /// Create a window
    namedWindow(window_name, CV_WINDOW_AUTOSIZE);

    /// Create a Trackbar for user to enter threshold
    createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

    /// Show the image
    CannyThreshold(0, nullptr);

    imwrite("edges.png", edges);
    waitKey(0);
    destroyWindow(window_name);
    namedWindow(window_name, CV_LOAD_IMAGE_COLOR);

    //-------------------------------------------------------------------------------
    // Segmentation
    if (display_caption(image, "segmentation") != 0) { return 0; }

    apply_segmentation(image);
    waitKey(0);

    //-------------------------------------------------------------------------------
    // Face detection
    if (display_caption(image, "face detection") != 0) { return 0; }

    return 0;
}


int display_caption(Mat src, char *caption) {
    Mat d;
    d = Mat::zeros(src.cols / 4, src.rows / 2, src.type());
    putText(d, caption,
            Point(10, src.rows / 8),
            CV_FONT_HERSHEY_COMPLEX, 1, Scalar(108, 200, 57));

    imshow(window_name, d);
    int c = waitKey(DELAY_CAPTION);
    if (c >= 0) { return -1; }
    return 0;
}

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void *) {
    /// Reduce noise with a kernel 3x3
    blur(grey_image, detected_edges, Size(3, 3));

    /// Canny detector
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);

    /// Using Canny's output as a mask, we display our result
    edges = Scalar::all(0);

    image.copyTo(edges, detected_edges);
    imshow(window_name, edges);
}

void apply_segmentation(Mat src) {
    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<Vec3b>(i, j) == Vec3b(255, 255, 255)) {
                src.at<Vec3b>(i, j)[0] = 0;
                src.at<Vec3b>(i, j)[1] = 0;
                src.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }

    // Create a kernel that we will use to sharpen our image
    Mat kernel =
            (Mat_<float>(3, 3)
                    <<
                    1, 1, 1,
                    1, -8, 1,
                    1, 1, 1);
    // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    // imshow( "New Sharped Image", imgResult );
    // Create binary image from source image
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    // imshow("Binary Image", bw);
    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    // imshow("Distance Transform Image", dist);
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    // imshow("Peaks", dist);
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
    }
    // Draw the background marker
    circle(markers, Point(5, 5), 3, Scalar(255), -1);
    // imshow("Markers", markers*10000);
    // Perform the watershed algorithm
    watershed(imgResult, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);
    // imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++) {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.emplace_back((uchar) b, (uchar) g, (uchar) r);
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size())) {
                dst.at<Vec3b>(i, j) = colors[index - 1];
            }
        }
    }
    // Visualize the final image
    imshow(window_name, dst);
}