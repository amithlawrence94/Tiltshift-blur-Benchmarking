#include <jni.h>
#include <string>
#include <cpu-features.h>
#include <thread>
#include <cmath>
#include <android/log.h>
#include <typeinfo>
#include <chrono>
using namespace std;
#include <arm_neon.h>
#include <iostream>
#include <cmath>
#include <android/log.h>
#include <jni.h>
#include <arm_neon.h>
#include <cmath>
int pixel_size;
jint* h;

//getSigma_neon function calculates sigma values for different values of y
float computeSigma_neon(int row, int a0, int a1, int a2, int a3, float32_t sigma_far, float32_t sigma_near){

    float multiplier = 0.0f;
    if (row<a0) { multiplier = sigma_far;}
    else if (row<a1) {multiplier = (sigma_far *(a1-row)+ 1.0f*(row-a0))/(a1-a0);}
    else if (row<=a2) {multiplier = 0.0f;}
    else if (row<a3) {multiplier = (1.0f*(a3-row)+ sigma_near*(row-a2))/(a3-a2);}
    else{ multiplier = sigma_near; }
    return multiplier;
}

//getK_neon function calculates kernel radius vector from sigma
int* CalK_neon(float sigma){
    int r=(int)ceil(3*sigma);
    int size=2*r+1;
    int* k=new int[size];             // k vector for G calculation
    int m=-r;
    for(int i=0; i<size ; i++){
        k[i]=m++;
    }
    return k;
}

//Calculates A R G B vector putting each of them in different lane from pixel pointer and return
uint32x4_t SeperateARGB(int* pixel){
    uint8_t* inputArrPtr = (uint8_t *)pixel;
    uint8x16x4_t pixelChannels = vld4q_u8(inputArrPtr);
    uint8x16_t A = pixelChannels.val[3];
    uint8x16_t R = pixelChannels.val[2];
    uint8x16_t G = pixelChannels.val[1];
    uint8x16_t B = pixelChannels.val[0];

    uint32x4_t ARGB = vdupq_n_u32(0);

    uint32_t AA = (uint32_t)vgetq_lane_u8(A,0);
    uint32_t RR = (uint32_t)vgetq_lane_u8(R,0);
    uint32_t GG = (uint32_t)vgetq_lane_u8(G,0);
    uint32_t BB = (uint32_t)vgetq_lane_u8(B,0);

    ARGB=vsetq_lane_u32(AA,ARGB,3);
    ARGB=vsetq_lane_u32(RR,ARGB,2);
    ARGB=vsetq_lane_u32(GG,ARGB,1);
    ARGB=vsetq_lane_u32(BB,ARGB,0);

    return ARGB;
}

// Function to calculate the shifted pixel arrays  and perform 1D-convolution
float32x4_t qVector_neon(int x, int y, int width, int* k, int klen, double* G, int* pixels, int length){
    //initialize a 4-lane vector with zeroes
    float32x4_t q=vdupq_n_f32(0);
    uint32x4_t p;

    for(int i=0; i<klen; i++){

        if((y+k[i])<0 || x<0 || (y+k[i])*width+x>= length) continue;//checking the conditions for the edge cases
        else {
            // Assign the pixel values
            p= SeperateARGB(&pixels[(y+k[i])*width+x]);
        }
        //initialize a vector to hold the gaussian weights
        float32x4_t Gvector = vdupq_n_f32((float)G[i]);
        float32x4_t p_temp = vcvtq_f32_u32(p);
        //multiply and add function
        //q = q + p_temp * Gvector
        q = vmlaq_f32(q, p_temp,Gvector);
    }
    return q;
}

//calculates the final weighted sum using the output of q vector
uint32x4_t pVector_neon(int x, int y, int width, int* k, int klen, double* G, int* pixel, int length){
    float32x4_t P1 = vdupq_n_f32(0);
    for(int i=0; i<klen; i++){
        float32x4_t GVector = vdupq_n_f32((float)G[i]);
        // Calculate the highest pixel to be accessed and verify if it is within the image dimensions
        if(y*width+x+k[i]<length)
            //Computed the new value by calling the qVector_neon function
            P1 = vmlaq_f32(P1, GVector, qVector_neon(x+k[i],y, width, k,klen, G, pixel, length));
    }
    uint32x4_t P = vcvtq_u32_f32(P1);
    return P;
}

// function to compute the gaussian weights using the indices assigned in the kernel variable as per the above function
double* GaussianWeightCal(int* k, float sigma,int ksize){
    double* weight = new double[ksize];
    for(int i=0; i < ksize; i++){
        //Determine the wights by plugging in values in pdf of gaussian distribution
        weight[i]=( exp( (-1 * k[i] * k[i] ) / (2 * sigma * sigma) ) / sqrt( 2 * M_PI * sigma * sigma) );
    }
    return weight;
}

// Then, a first transformation generates the intermediate matrix, q(y, x):
float qVector(string channel, int x, int y, int width, int* k, double* G, int* pixels,int ksize) {
    double q = 0;
    int p;
    //The following switch-case based on the color AA RR BB GG
    if (channel == "BB") {
        for (int i = 0; i < ksize; i++) {
            //zero padding for edges
            if ((y + k[i]) < 0 || x < 0 || (y + k[i]) * width + x >= pixel_size) { p = 0; }//checking the conditions for the edge cases
            else { p = pixels[(y + k[i]) * width + x]; }// Assign the pixel values
            // Isolate the blue pixel value from the original pixel value
            int BB = p & 0xff;
            q += BB * G[i];
        }
    }
    else if (channel == "GG") {
        for (int i = 0; i < ksize; i++) {
            if ((y + k[i]) < 0 || x < 0 || (y + k[i]) * width + x >= pixel_size) { p = 0; }//checking the conditions for the edge cases
            else { p = pixels[(y + k[i]) * width + x]; }// Assign the pixel values
            // make a 8 bit right shift
            // Isolate the blue pixel value from the original pixel value
            int GG = (p >> 8) & 0xff;
            q += GG * G[i];
        }
    }
    else if (channel == "RR"){
        for (int i = 0; i < ksize; i++) {
            if ((y + k[i]) < 0 || x < 0 || (y + k[i]) * width + x >= pixel_size) { p = 0; }//checking the conditions for the edge cases
            else { p = pixels[(y + k[i]) * width + x]; }// Assign the pixel values
            // make a 8 bit right shift
            // Isolate the blue pixel value from the original pixel value
            int RR = (p >> 16) & 0xff;
            q += RR * G[i];
        }
    }
    else if (channel == "AA"){
        for(int i=0; i< ksize; i++){
            if((y + k[i]) < 0 || x<0 || (y + k[i]) * width + x >= pixel_size){p = 0;}//checking the conditions for the edge cases
            else {p = pixels[( y + k[i] ) * width + x]; }// Assign the pixel values
            // make a 8 bit right shift
            // Isolate the blue pixel value from the original pixel value
            int AA = (p>>24) &0xff;
            q += AA * G[i];
        }
    }
    else{
        __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_C++_NO CHANNEL", "Good Day");
    }
    return q;
}

//calculates the final weighted sum using the output of q vector
int pVector(string channel, int x, int y, int width, int* k, double* G, int* pixel,int ksize){
    double P=0;
    for(int i=0; i < ksize; i++){
        // Calculate the highest pixel to be accessed and verify if it is within the image dimensions
        if( y * width + x + k[i] < pixel_size) {
            //Computed the new value by calling the qVector function
            P = P + G[i] * qVector(channel, x + k[i], y, width, k, G, pixel, ksize);
        }
    }
    return (int)P ;
}

// Thread Manipulation function
void Sigma_far_near(string threadName,float sigma,int a0,int a1,int a2,int a3,int* pixels,int height,int width) {
    __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_C++_THREAD",
            "creating Thread:%s,Sigma:%f,a0:%d,a1:%d,a2:%d,a3:%d",&threadName,sigma,a0,a1,a2,a3);
    //Determine the current region in the image by comparing the threadname variable
    //
    //Region 1 - between row 0 and a0
    //
    //In the sigma_far region between the starting index and a0, the sigma value remains unchanged
    if (threadName == "Thread-sigma-far-solid") {
        if (sigma >= 0.6){
            // Determine the Kernel radius and assign indices using the sigma value
            int r = (int) ceil(2 * sigma);
            int kernel_size = 2 * r + 1;
            int* k = new int[kernel_size];             // initializing Kernel vector
            int m = -r;  // set initail point to -r
            // loop over the entire kernel and initialize as per increasing decreasing index
            for (int i = 0; i < kernel_size; i++) {
                k[i] = m++;
            }
            //Determine the gaussian weights using the kernel indices and the sigma value
            double* kernelMatrix = GaussianWeightCal(k, sigma,kernel_size);
            for (int i=0; i < a0; i++){
                //Iterating through the image row by row
                //Compute the result of convolution with Gaussian kernel for each R,G,B,A color channel
                for(int j=0; j < width; j++){
                    int pBB = pVector("BB", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    int pRR = pVector("RR", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    int pGG = pVector("GG", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    int pAA = pVector("AA", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    //cout("pBB:%d ,pRR:%d ,pGG:%d ,pAA:%d",pBB,pRR,pGG,pAA);
                    // Copy the new pixel values to pixelsOut
                    h[i * width + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                }
            }
        }
            //
            // If sigma is below the  threshold value the pixels are left untouched
            //
        else {
            for (int i = 0; i < a0; i++) {
                // Copy the new pixel values to pixelsOut
                for (int j = 0; j < width; j++) {
                    h[i * width + j] = pixels[i * width + j];
                }
            }
        }
    }
        //
        //Region 2 - between row a0 and a1
        //
        //In this region the sigma value gradually drops off till it dips below the threshold value
    else if(threadName == "Thread-sigma-far-gradual"){
        float sigmaY;
        for (int i = a0 ; i < a1 ; i++) {
            //Determining the steadily decreasing sigma value
            sigmaY = sigma * ((float) (a1-i)/(a1-a0));
            if(sigmaY >= 0.6) {
                //Determine the Kernel radius and assign indices using the decreasing sigma values
                int r = (int) ceil(2 * sigmaY);
                int kernel_size = 2 * r + 1;
                int* k = new int[kernel_size];             // initializing Kernel vector
                int m = -r;  // set initail point to -r
                // loop over the entire kernel and initialize as per increasing decreasing index
                for (int i = 0; i < kernel_size; i++) {
                    k[i] = m++;
                }
                //Determine the gaussian weights using the kernel indices and the sigma value
                double* kernelMatrix = GaussianWeightCal(k, sigmaY,kernel_size);  //Calculate the Gaussian Matrix
                // Convolution for all the channels with the gaussian weight matrix
                for (int j = 0; j < width; j++) {
                    int pBB = pVector("BB", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    int pRR = pVector("RR", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    int pGG = pVector("GG", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    int pAA = pVector("AA", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    // Copy the new pixel values to pixelsOut
                    h[i * width + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                }
            }
                //
                // If sigma is below the  threshold value the pixels are left untouched
                //
            else{
                for(int z = i ; z <= a1 ; z++){
                    // Copy the new pixel values to pixelsOut
                    for (int j = 0; j < width; j++) {
                        h[z * width + j] = pixels[z * width + j];
                    }
                }
                break;
            }
        }
    }
        //
        //Region 3 - between row a1 and a2
        //
        //In this region the image pixels are left untouched
    else if (threadName == "Thread-sigma-no-blur") {
        for (int j = a1; j <= a2; j++) {
            //Iterating through the image row by row
            for (int i = 0; i < width; i++) {
                //Copy pixels from original image to the pixelsOut
                h[j * width + i] = pixels[j * width + i];
            }
        }
    }
        //
        //Region 4 - between row a2 and a3
        //
        //In this region the sigma value gradually increase till it rises above the threshold value
    else if(threadName == "Thread-sigma-near-gradual") {
        float sigmaX;
        for (int i = a3; i > a2; i--) {
            //Determining the steadily increasing sigma value
            sigmaX = sigma * ((float)(i-a2)/(a3-a2));
            if (sigmaX >= 0.6) {
                //Determine the Kernel radius and assign indices using the increasing sigma values
                int r = (int) ceil(2 * sigmaX);
                int kernel_size = 2 * r + 1;
                int* k = new int[kernel_size];             // initializing Kernel vector
                int m = -r;  // set initail point to -r
                // loop over the entire kernel and initialize as per increasing decreasing index
                for (int i = 0; i < kernel_size; i++) {
                    k[i] = m++;
                }
                //Determine the gaussian weights using the kernel indices and the sigma value
                double* kernelMatrix = GaussianWeightCal(k, sigmaX,kernel_size);  //Calculate the Gaussian Matrix
                // Convolution for all the channels with the gaussian weight matrix
                for (int j = 0; j < width; j++) {
                    //Compute the result of convolution with Gaussian kernel for each R,G,B,A color channel
                    int pBB = pVector("BB", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    int pRR = pVector("RR", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    int pGG = pVector("GG", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    int pAA = pVector("AA", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    // Copy the new pixel values to pixelsOut
                    h[i * width + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                }
            }
                //
                // If sigma is below the  threshold value the pixels are left untouched
                //
            else {
                for(int z = i; z > a2 ; z--) {
                    for (int j = 0; j < width; j++) {
                        // Copy the new pixel values to pixelsOut
                        h[z * width + j] = pixels[z * width + j];
                    }
                }
                break;
            }
        }
    }
        //
        //Region 5 - between row a3 and final
        //
        //In the sigma_far region between the a3 and final value, the sigma value remains unchanged
    else if(threadName == "Thread-sigma-near-solid") {
        if (sigma >= 0.6) {
            //Determine the Kernel radius and assign indices using the sigma value
            int r = (int) ceil(2 * sigma);
            int kernel_size = 2 * r + 1;
            int* k = new int[kernel_size];             // initializing Kernel vector
            int m = -r;  // set initail point to -r
            // loop over the entire kernel and initialize as per increasing decreasing index
            for (int i = 0; i < kernel_size; i++) {
                k[i] = m++;
            }
            //Determine the gaussian weights using the kernel indices and the sigma value
            double *kernelMatrix = GaussianWeightCal(k, sigma,kernel_size);
            for (int i = a3; i < height; i++) {
                //Iterating through the image row by row
                // Convolution for all the channels with the gaussian weight matrix
                for (int j = 0; j < width; j++) {
                    int pBB = pVector("BB", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    int pRR = pVector("RR", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    int pGG = pVector("GG", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    int pAA = pVector("AA", j, i, width, k, kernelMatrix, pixels,kernel_size);
                    // Copy the new pixel values to pixelsOut
                    h[i * width + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 |
                                       (pBB & 0xff);
                }
            }
        }
            //
            // If sigma is below the  threshold value the pixels are left untouched
            //
        else {
            for (int i = a3; i < height; i++) {
                // Copy the new pixel values to pixelsOut
                for (int j = 0; j < width; j++) {
                    h[i * width + j] = pixels[i * width + j];
                }
            }
        }
    }
}

extern "C"
JNIEXPORT jint JNICALL
Java_edu_asu_ame_meteor_speedytiltshift2018_SpeedyTiltShift_tiltshiftcppnative(JNIEnv *env,
                                                                               jobject instance,
                                                                               jintArray inputPixels_,
                                                                               jintArray outputPixels_,
                                                                               jint width,
                                                                               jint height,
                                                                               jfloat sigma_far,
                                                                               jfloat sigma_near,
                                                                               jint a0, jint a1,
                                                                               jint a2, jint a3) {
//    Get the input and output pixels from the java environment
    jint *pixels = env->GetIntArrayElements(inputPixels_, NULL);
    jint *outputPixels = env->GetIntArrayElements(outputPixels_, NULL);
    // Initialize the temp pixels array and pixels size for manipulation
    h = new jint[height*width];
    pixel_size = height*width;

    // just a feature for multiplying the sigma value
    int scale = 1;
    __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_C++_MAIN","height:%d, width:%d ,pixel_size:%d",height,width,height*width);
    __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_C++_MAIN",
            "sigma far:%f,sigma near:%f,a0:%d,a1:%d,a2:%d,a3:%d",sigma_far*scale,sigma_near*scale,a0,a1,a2,a3);
    // try catch block to call the below 5 threads and join them
    // A. SigmaFarSolid - from 0 to a0 with constant sigma
    // B. SigmaFarGradual - from a0 to a1 with varying sigma
    // C. SigmaNoBlur - from a1 to a2
    // D. SigmaNearGradual - from a2 to a3 with varying sigma
    // E. SigmaNearSolid - from a3 to end of the height constant sigma
    try {
        std::thread SigmaFarSolid(Sigma_far_near, "Thread-sigma-far-solid", sigma_far * scale, a0,a1, a2, a3, pixels, height, width);
        std::thread SigmaFarGradual(Sigma_far_near, "Thread-sigma-far-gradual", sigma_far * scale,a0, a1, a2, a3, pixels, height, width);
        std::thread SigmaNoBlur(Sigma_far_near, "Thread-sigma-no-blur", sigma_far, a0, a1, a2, a3,pixels, height, width);
        std::thread SigmaNearGradual(Sigma_far_near, "Thread-sigma-near-gradual",sigma_near * scale, a0, a1, a2, a3, pixels, height, width);
        std::thread SigmaNearSolid(Sigma_far_near, "Thread-sigma-near-solid", sigma_near * scale,a0, a1, a2, a3, pixels, height, width);

        SigmaFarSolid.join();
        SigmaFarGradual.join();
        SigmaNoBlur.join();
        SigmaNearGradual.join();
        SigmaNearSolid.join();
    }
    catch (std::exception& e)
    {
        __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_C++_THREAD TRY","%s",e.what());
    }
    // copy the image from the temp to outputPixels
//    for (int i = 0; i < height ; i++) {
//        for (int j = 0; j < width; j++) {
//            outputPixels[i * width + j] = h[i * width + j];
//        }
//    }
    env->ReleaseIntArrayElements(inputPixels_, pixels, 0);
    env->ReleaseIntArrayElements(outputPixels_, h, 0);
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_edu_asu_ame_meteor_speedytiltshift2018_SpeedyTiltShift_tiltshiftneonnative(JNIEnv *env,
                                                                                jclass instance,
                                                                                jintArray inputPixels_,
                                                                                jintArray outputPixels_,
                                                                                jint width,
                                                                                jint height,
                                                                                jfloat sigma_far,
                                                                                jfloat sigma_near,
                                                                                jint a0, jint a1,
                                                                                jint a2, jint a3) {
//Parse elements from JNI Array to use in native code

    jint *pixels = env->GetIntArrayElements(inputPixels_, NULL);
    jint *outputPixels = env->GetIntArrayElements(outputPixels_, NULL);
    long pixel_length = env->GetArrayLength(inputPixels_);
    __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_NEON_MAIN","height:%d, width:%d ,pixel_size:%d",height,width,height*width);
    __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_NEON_MAIN", "length: %d %d %d %d %d",pixel_length,a0,a1,a2,a3);
    int flag = 0, count = 0,a4 = 0;
    for (int y= 0; y < height; y++){
        float sigma =0;
        //Iterating through the image row by row
        sigma=computeSigma_neon(y,a0,a1,a2,a3,sigma_far,sigma_near);
        if(sigma<=0.6){
            if (flag == 0) {
                a4 = y;
                flag = 1;
            }
            count++;
            continue;
        }
        int r = (int)ceil(2.5*sigma);
        int kernel_size = 2*r+1;
        int* k= new int[kernel_size];
        //Determine the Kernel radius and assign indices using the sigma value
        k= CalK_neon(sigma);
        //Determine the gaussian weights using the kernel indices and the sigma value
        double* kernelMatrix = new double[kernel_size];
        kernelMatrix = GaussianWeightCal(k, sigma, kernel_size);
        for (int x = 0; x < width; x++) {
            // Iterating through the image row by row
            // Convolution for all the channels with the gaussian weight matrix
            uint32x4_t ARGB = pVector_neon(x, y, width, k, kernel_size, kernelMatrix, pixels, pixel_length);
            int partA = vgetq_lane_u32(ARGB, 3);
            int partR = vgetq_lane_u32(ARGB, 2);
            int partG = vgetq_lane_u32(ARGB, 1);
            int partB = vgetq_lane_u32(ARGB, 0);
            // Copy the new pixel values to pixelsOut
            outputPixels[y * width + x] = (partA & 0xff) << 24 | (partR & 0xff) << 16 | (partG & 0xff) << 8 | (partB & 0xff);
        }
//        free array memory
        delete[] k;
        delete[] kernelMatrix;
    }
    //
    // If sigma is below the  threshold value the pixels are left untouched
    //
    for (int j = a4;j <= a4+count; j++) {
        // Copy the new pixel values to pixelsOut
        for (int i=0; i<width ;i++) {
            outputPixels[j * width + i] = pixels[j * width + i];
        }
    }
    env->ReleaseIntArrayElements(inputPixels_, pixels, 0);
    env->ReleaseIntArrayElements(outputPixels_, outputPixels, 0);
    return 0;
}