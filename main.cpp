#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv/cv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/// Global Variables
const int DELAY_CAPTION = 500;
const int MAX_KERNEL_LENGTH = 31;
const char window_name[] = {'D', 'i', 's', 'p', 'l', 'a', 'y', ' ', 'w', 'i', 'n', 'd', 'o', 'w'};
int lowThreshold;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
Mat image, edges, grey_image, detected_edges;

void do_framing(char **argv);

void do_face_detection(char **argv);

///functions
void CannyThreshold(int, void *);

int display_caption(Mat, char *);

void extract_blue_channel();

void convert_to_greyscale();

void applying_gaussian_blur();

void do_rotate();

void do_resize();

void do_edge_detection();

void detectAndDraw(Mat, CascadeClassifier,
                   const CascadeClassifier &,
                   double);

void apply_segmentation(Mat);


//input : image , video , classifier 1 , classifier 2
//input example: C:\Users\amirphl\CLionProjects\Imgae_Processing_Project_Phase_1\Lenna.png C:\Users\amirphl\Desktop\videos\1.mp4 C:\opencv\mingw-build\install\etc\haarcascades\\haarcascade_fullbody.xml C:\opencv\mingw-build\install\etc\haarcascades\haarcascade_frontalface_default.xml
int main(int argc, char **argv) {
    if (argc != 5) {
        cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }
    char arr_1[] = "Could not open or find the image...";
    char arr_2[] = "image";
    char arr_3[] = "blue channel";
    char arr_4[] = "greyscale";
    char arr_5[] = "Gaussian blur";
    char arr_6[] = "rotated image";
    char arr_7[] = "resize";
    char arr_8[] = "detect edges";
    char arr_9[] = "segmentation";
    char arr_10[] = "face detection";
    char arr_11[] = "farming";

    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    resize(image, image, Size(1500, 1100), 0, 0);
    if (!image.data) {
        if (display_caption(image, arr_1) != 0) { return 0; }
        return -1;
    }
    namedWindow(window_name, WINDOW_AUTOSIZE);
    if (display_caption(image, arr_2) != 0) { return 0; }
    imshow(window_name, image);
    waitKey(0);

    if (display_caption(image, arr_3) != 0) { return 0; }
    extract_blue_channel();

    if (display_caption(image, arr_4) != 0) { return 0; }
    convert_to_greyscale();

    if (display_caption(image, arr_5) != 0) { return 0; }
    applying_gaussian_blur();

    if (display_caption(image, arr_6) != 0) { return 0; }
    do_rotate();

    if (display_caption(image, arr_7) != 0) { return 0; }
    do_resize();

    if (display_caption(image, arr_8) != 0) { return 0; }
    do_edge_detection();

    if (display_caption(image, arr_9) != 0) { return 0; }
    apply_segmentation(image);

    if (display_caption(image, arr_10) != 0) { return 0; }
    do_face_detection(argv);

    if (display_caption(image, arr_11) != 0) { return 0; }
    do_framing(argv);

    destroyAllWindows();
    return 0;
}

void extract_blue_channel() {
    Mat temp = image.clone();
    for (int j = 0; j < image.rows; ++j) {
        for (int i = 0; i < image.cols; ++i) {
            Vec3b color = temp.at<Vec3b>(Point(i, j));
            color[1] = 0;
            color[2] = 0;
            temp.at<Vec3b>(Point(i, j)) = color;
        }
    }

    //Note: OpenCV uses BGR color order
    imwrite("blue.png", temp);
    imshow(window_name, temp);
    waitKey(0);
}

void convert_to_greyscale() {
    cvtColor(image, grey_image, CV_BGR2GRAY);
    imwrite("greyscale.png", grey_image);
    imshow(window_name, grey_image);
    waitKey(0);
}


void applying_gaussian_blur() {
    Mat smoothed_image;
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
        GaussianBlur(image, smoothed_image, Size(i, i), 0, 0);
    }
    imwrite("Gaussian_blur.png", smoothed_image);
    imshow(window_name, smoothed_image);
    waitKey(0);
}

void do_rotate() {
    Mat rotated_image, smoothed_image;
    const float angle = 90;
    // get rotation matrix for rotating the image around its center in pixel coordinates
    Point2f center((image.cols - 1) / 2.0, (image.rows - 1) / 2.0);
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
}

void do_edge_detection() {
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
}

void do_resize() {
    Mat resized_image;
    const float m_x = 0.5;
    const float m_y = 1;
    resize(image, resized_image, Size(), m_x, m_y);
    imwrite("resized_image.png", resized_image);
    imshow(window_name, resized_image);
    waitKey(0);
}

void do_face_detection(char **argv) {
    // PreDefined trained XML classifiers with facial features
    CascadeClassifier cascade, nestedCascade;
    double scale = 1;
    // Load classifiers from "opencv/data/haarcascades" directory
//    nestedCascade.load(R"(C:\opencv\mingw-build\install\etc\haarcascades\haarcascade_fullbody.xml)");
    nestedCascade.load(argv[3]);
    // Change path before execution
//    cascade.load(R"(C:\opencv\mingw-build\install\etc\haarcascades\haarcascade_frontalface_default.xml)");
    cascade.load(argv[4]);
    detectAndDraw(image.clone(), cascade, nestedCascade, scale);
    waitKey(0);
}

void do_framing(char **argv) {
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap(argv[2]);
    const int num_of_frames = 5;
    const int delay = 500; //ms
    int counter = 0;
    while (counter < num_of_frames) {
        Mat frame;
        // Capture frame-by-frame
        cap >> frame;
        // If the frame is empty, break immediately
        if (frame.empty())
            break;
        // Display the resulting frame
        imshow("video", frame);
        waitKey(delay);
        counter++;
    }
    // When everything done, release the video capture object
    cap.release();
}


int display_caption(Mat src, char *caption) {
    Mat d = Mat::zeros(src.cols / 4, src.rows / 2, src.type());
    putText(d, caption, Point(10, src.rows / 8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(108, 200, 57));
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
    imshow("Laplace Filtered Image", imgLaplacian);
    imshow("New Sharped Image", imgResult);
    // Create binary image from source image
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("Binary Image", bw);
    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    imshow("Peaks", dist);
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
    imshow("Markers", markers * 10000);
    // Perform the watershed algorithm
    watershed(imgResult, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);
    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
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
    imwrite("segmented.png", dst);
    imshow(window_name, dst);
    waitKey(0);
}

void detectAndDraw(Mat img, CascadeClassifier cascade,
                   const CascadeClassifier &nestedCascade,
                   double scale) {
    vector<Rect> faces;
    Mat gray, smallImg;

    double fx = 1 / scale;
    cvtColor(img, gray, COLOR_BGR2GRAY); // Convert to Gray Scale
    // Resize the Grayscale Image
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);

    equalizeHist(smallImg, smallImg);
    // Detect faces of different sizes using cascade classifier
    cascade.detectMultiScale(smallImg, faces, 1.1,
                             2, 0 | CASCADE_SCALE_IMAGE, Size(10, 20));

    // Draw rectangles around the faces
    for (int i = 0; i < faces.size(); i++) {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Scalar color = Scalar(244, 66, 101); // Color for Drawing tool

        rectangle(img, cvPoint(cvRound(faces[i].x * scale), cvRound(faces[i].y * scale)),
                  cvPoint(cvRound((faces[i].x + faces[i].width - 1) * scale),
                          cvRound((faces[i].y + faces[i].height - 1) * scale)), color, 3, 8, 0);
        if (nestedCascade.empty())
            continue;
    }

    // Show Processed Image with detected faces
    imwrite("face.png", img);
    imshow(window_name, img);
}
