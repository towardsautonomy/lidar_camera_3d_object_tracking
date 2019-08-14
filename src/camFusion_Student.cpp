
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-100, bottom+50), cv::FONT_ITALIC, 1, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-100, bottom+90), cv::FONT_ITALIC, 1, currColor); 
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for(auto it: kptMatches) 
    {
        if(boundingBox.roi.contains(cv::Point(kptsCurr[it.queryIdx].pt.x, kptsCurr[it.queryIdx].pt.y))) {
            boundingBox.keypoints.push_back(kptsCurr[it.queryIdx]);
            boundingBox.kptMatches.push_back(it);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // find mean of keypoints
    cv::KeyPoint meanKptCurr, meanKptPrev;
    meanKptCurr.pt.x = 0;
    meanKptCurr.pt.y = 0;
    meanKptPrev.pt.x = 0;
    meanKptPrev.pt.y = 0;
    for(auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        meanKptCurr.pt.x += kptsCurr[(*it).queryIdx].pt.x;
        meanKptCurr.pt.y += kptsCurr[(*it).queryIdx].pt.y;
        meanKptPrev.pt.x += kptsPrev[(*it).trainIdx].pt.x;
        meanKptPrev.pt.y += kptsPrev[(*it).trainIdx].pt.y;
    }
    meanKptCurr.pt.x /= kptMatches.size();
    meanKptCurr.pt.y /= kptMatches.size();
    meanKptPrev.pt.x /= kptMatches.size();
    meanKptPrev.pt.y /= kptMatches.size();

    std::vector<double> distRatios;
    for(auto it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1)
    {
        cv::KeyPoint kptOuterCurr = kptsCurr[(*it1).queryIdx];
        cv::KeyPoint kptOuterPrev = kptsPrev[(*it1).trainIdx];
        for(auto it2 = kptMatches.begin(); it2 != kptMatches.end(); ++it2)
        {
            if(it1 != it2)
            {
                cv::KeyPoint kptInnerCurr = kptsCurr[(*it2).queryIdx];
                cv::KeyPoint kptInnerPrev = kptsPrev[(*it2).trainIdx];

                double distCurr = cv::norm(kptOuterCurr.pt - kptInnerCurr.pt);
                double distPrev = cv::norm(kptOuterPrev.pt - kptInnerPrev.pt);

                double minDist = 50.0;
                double max_dist_from_mean = 150.0;
                // avoid divide by zero and only consider keypoints further apart
                if((distPrev > std::numeric_limits<double>::epsilon()) &&
                   (distCurr > minDist) &&
                   (distCurr < max_dist_from_mean)) {
                    distRatios.push_back(distCurr/distPrev);
                }
            }
        }
    }

    // make sure that the size of distRatios is greater than 0
    if(distRatios.size() == 0)
    {
        TTC = NAN;
    }
    else
    {
        // use mean distance ratio
        std::sort(distRatios.begin(), distRatios.end());
        int medIdx = distRatios.size() / 2;
        double medDistRatio;
        if((distRatios.size() % 2) == 0) {
            medDistRatio = (distRatios[medIdx - 1] + distRatios[medIdx])/2.0;
        }
        else {
            medDistRatio = distRatios[medIdx];
        }
        double delta_t = 1.0/frameRate;
        TTC = -1.0*delta_t / (1 - medDistRatio);
    }
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    float minX_prev =1e9, minX_curr = 1e9;
    float avgX_prev = 0, avgX_curr = 0;
    float max_dist_tol = 1.0;
    for(auto it: lidarPointsPrev) {
        avgX_prev += it.x;
        if(it.x < minX_prev) minX_prev = it.x;
    }
    avgX_prev /= lidarPointsPrev.size();

    for(auto it: lidarPointsCurr) {
        avgX_curr += it.x;
        if(it.x < minX_curr) minX_curr = it.x;
    }
    avgX_curr /= lidarPointsCurr.size();

    float delta_t = 1.0/frameRate;
    // use average distance to compute the relative velocity
    float rel_vel = (avgX_curr - avgX_prev)/delta_t;

    // use closest distance to compute TTC only if it is close to the mean
    // if the point is not closer to the mean that would imply that it is erroneous
    if(fabs(avgX_curr - minX_curr) < max_dist_tol) {
        TTC = -1.0*minX_curr/rel_vel;
    }
    else {
        TTC = -1.0*avgX_curr/rel_vel;
    }
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    const int min_n_keypoints = 4;
    DataFrame prevFrameCpy = prevFrame;
    for(auto it1=currFrame.boundingBoxes.begin(); it1!=currFrame.boundingBoxes.end(); ++it1)
    {
        std::vector<int> n_keypoints;
        for(auto it2=prevFrameCpy.boundingBoxes.begin(); it2!=prevFrameCpy.boundingBoxes.end(); ++it2)
        {
            int n = 0;
            for(auto i : matches)
            {
                if(((*it1).roi.contains(cv::Point(currFrame.keypoints[i.queryIdx].pt.x,currFrame.keypoints[i.queryIdx].pt.y))) &&
                   ((*it2).roi.contains(cv::Point(prevFrameCpy.keypoints[i.trainIdx].pt.x, prevFrame.keypoints[i.trainIdx].pt.y))))
                {
                    n++;
                }
            }
            n_keypoints.push_back(n);
        }

        // check if there are at least 'min_n_keypoints' keypoints contained within the ROI
        if(n_keypoints.size() > min_n_keypoints) {
            int max_match_idx = std::distance(n_keypoints.begin(), std::max_element(n_keypoints.begin(), n_keypoints.end()));
            bbBestMatches.insert(std::pair<int,int>(prevFrameCpy.boundingBoxes[max_match_idx].boxID, (*it1).boxID));
            
            // remove that bounding-box from previous to avoid multiple matches
            //prevFrameCpy.boundingBoxes.erase(prevFrameCpy.boundingBoxes.begin() + max_match_idx);
        }
    }
}
