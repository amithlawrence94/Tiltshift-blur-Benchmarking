package edu.asu.ame.meteor.speedytiltshift2018;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import java.io.* ;
import android.widget.TextView;
import android.os.*;
import java.lang.*;

public class SpeedyTiltShift {
    static SpeedyTiltShift Singleton = new SpeedyTiltShift();
    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }
    // Initialize the timer to record time elapsed
    public static long javaElapsedTime=0;
    // Extending thread class function and define the class variables
    // Define the operations to be performed on the variables depending on the threadname invoked
    public static class Sigma_far_near extends Thread {
        private static int imgHeight,imgWidth;
        private int offset0,offset1,offset2,offset3;
        private float sigma;
        private int[] pixels;
        public  static int[] pixelsOut;
        private String threadName;
        //Define the variables to be passed to each thread
        Sigma_far_near( String name, float Sigma,int a0,int a1,int a2,int a3,int[] pixel,int imHeight,int imWidth) {
            threadName = name;
            offset0 = a0;
            offset1 = a1;
            offset2 = a2;
            offset3 = a3;
            imgHeight = imHeight;
            imgWidth = imWidth;
            pixels = pixel;
            pixelsOut = pixels;
            sigma = Sigma;
            Log.d("TILTSHIFT_JAVA_THREAD","Creating "
                    + threadName +",a0:" + offset0 +",a1:" + offset1+",a2:" + offset2+",a3:" + offset3
                    +",H:" + imgHeight +",W:" + imgWidth
                    +",Sigma:"+ Sigma +",input Pixels length:" +pixels.length
                    +",O/p Pixel:"+pixelsOut.length);
        }
        //Define action to be performed for thread
        public void run()
        {
            //Log.d("TILTSHIFT_JAVA_THREAD", "Thread " + Thread.currentThread().getId() + " is running with," + ",a0:" + offset0 +",a1:" + offset1+",a2:" + offset2+",a3:" + offset3 + "," + imgHeight + "," + imgWidth + "," + sigma);
            try
            {
                //Determine the current region in the image by comparing the threadname variable
                //
                //Region 1 - between row 0 and a0
                //
                //In the sigma_far region between the starting index and a0, the sigma value remains unchanged
                if (threadName == "Thread-sigma-far-solid") {
                    if(sigma >= 0.6){
                    //Log.d("TILTSHIFT_JAVA_THREAD","Inside "+threadName);
                    // Determine the Kernel radius and assign indices using the sigma value
                    int[] k= calK(sigma);
                    //Determine the gaussian weights using the kernel indices and the sigma value
                    double[] kernelMatrix = GaussianWeightCal(k, sigma);
                    for (int i=0; i<=offset0; i++){
                        //Iterating through the image row by row
                        //Compute the result of convolution with Gaussian kernel for each R,G,B,A color channel
                        for(int j=0; j<imgWidth; j++){
                            int pBB = pVector("BB", j, i, imgWidth, k, kernelMatrix, pixels);
                            int pRR = pVector("RR", j, i, imgWidth, k, kernelMatrix, pixels);
                            int pGG = pVector("GG", j, i, imgWidth, k, kernelMatrix, pixels);
                            int pAA = pVector("AA", j, i, imgWidth, k, kernelMatrix, pixels);
                            // Copy the new pixel values to pixelsOut
                            pixelsOut[i * imgWidth + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                        }
                    }
                    //Log.d("TILTSHIFT_JAVA_THREAD", "Return from Thread " + Thread.currentThread().getId() + "Length of pixelsOut is " + pixelsOut.length);
                }
                    //
                    // If sigma is below the  threshold value the pixels are left untouched
                    //
                    else{
                        for (int i=0; i<=offset0; i++){
                            // Copy the new pixel values to pixelsOut
                            for(int j=0; j<imgWidth; j++){
                                pixelsOut[i * imgWidth + j] = pixels[i * imgWidth + j];
                            }}
                    }
                }
                //
                //Region 5 - between row a3 and final
                //
                //In the sigma_far region between the a3 and final value, the sigma value remains unchanged
                else if(threadName == "Thread-sigma-near-solid"){
                    if(sigma >= 0.6){
                        //Determine the Kernel radius and assign indices using the sigma value
                        int[] k= calK(sigma);
                        //Determine the gaussian weights using the kernel indices and the sigma value
                        double[] kernelMatrix = GaussianWeightCal(k, sigma);
                        //Log.d("TILTSHIFT_JAVA_THREAD","Inside "+threadName);
                        for (int i=offset3; i < imgHeight; i++) {
                            // Iterating through the image row by row
                            // Convolution for all the channels with the gaussian weight matrix
                            for (int j = 0; j < imgWidth; j++) {
                                //Compute the result of convolution with Gaussian kernel for each R,G,B,A color channel
                                int pBB = pVector("BB", j, i, imgWidth, k, kernelMatrix, pixels);
                                int pRR = pVector("RR", j, i, imgWidth, k, kernelMatrix, pixels);
                                int pGG = pVector("GG", j, i, imgWidth, k, kernelMatrix, pixels);
                                int pAA = pVector("AA", j, i, imgWidth, k, kernelMatrix, pixels);
                                // Copy the new pixel values to pixelsOut
                                pixelsOut[i * imgWidth + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                            }
                        }
                        }
                    //
                    // If sigma is below the  threshold value the pixels are left untouched
                    //
                    else {
                        for (int i = offset3; i <= imgHeight; i++) {
                            // Copy the new pixel values to pixelsOut
                            for (int j = 0; j < imgWidth; j++) {
                                pixelsOut[i * imgWidth + j] = pixels[i * imgWidth + j];
                            }
                        }
                    }
                    //Log.d("TILTSHIFT_JAVA_THREAD", "Return from Thread " + Thread.currentThread().getId() + "Length of pixelsOut is" + pixelsOut.length);
                }
                //
                //Region 3 - between row a1 and a2
                //
                //In this region the image pixels are left untouched
                else if (threadName == "Thread-sigma-no-blur") {
                    for (int i = offset1; i < offset2 ; i++) {
                        //Iterating through the image row by row
                        for (int j = 0; j < imgWidth; j++) {
                            //Copy pixels from original image to the pixelsOut
                            pixelsOut[i * imgWidth + j] = pixels[i * imgWidth + j];
                        }
                    }
                    //Log.d("TILTSHIFT_JAVA_THREAD", "Return from Thread " + Thread.currentThread().getId() + "Length of pixelsOut is" + pixelsOut.length);
                }
                //
                //Region 2 - between row a0 and a1
                //
                //In this region the sigma value gradually drops off till it dips below the threshold value
                else if(threadName == "Thread-sigma-far-gradual") {
                    float sigmaY;
                    for (int i = offset0; i < offset1 ; i++) {
                        //Determining the steadily decreasing sigma value
                        sigmaY= sigma*((float) (offset1-i)/(offset1-offset0));
                        if(sigmaY >= 0.6) {
                            //Determine the Kernel radius and assign indices using the decreasing sigma values
                            int[] k = calK(sigmaY);                        //Calculating radius vector
                            //Determine the gaussian weights using the kernel indices and the sigma value
                            double[] kernelMatrix = GaussianWeightCal(k, sigmaY);  //Calculate the Gaussian Matrix

                            // Convolution for all the channels with the gaussian weight matrix
                            for (int j = 0; j < imgWidth; j++) {
                                //Compute the result of convolution with Gaussian kernel for each R,G,B,A color channel
                                int pBB = pVector("BB", j, i, imgWidth, k, kernelMatrix, pixels);
                                int pRR = pVector("RR", j, i, imgWidth, k, kernelMatrix, pixels);
                                int pGG = pVector("GG", j, i, imgWidth, k, kernelMatrix, pixels);
                                int pAA = pVector("AA", j, i, imgWidth, k, kernelMatrix, pixels);
                                // Copy the new pixel values to pixelsOut
                                pixelsOut[i * imgWidth + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                            }
                        }
                        //
                        // If sigma is below the  threshold value the pixels are left untouched
                        //
                        else{
                            for(int z = i ; z < offset1 ; z++){
                                // Copy the new pixel values to pixelsOut
                                for (int j = 0; j < imgWidth; j++) {
                                    pixelsOut[z * imgWidth + j] = pixels[z * imgWidth + j];
                                }
                            }
                            break;
                        }
                    }
                }
                //
                //Region 4 - between row a2 and a3
                //
                //In this region the sigma value gradually increase till it rises above the threshold value
                else if(threadName == "Thread-sigma-near-gradual") {
                    //Log.d("TILTSHIFT_JAVA_THREAD", "Sigma"+sigma);
                    float sigmaX;
                    for (int i = offset3; i > offset2; i--) {
                        sigmaX = sigma * ((float)(i-offset2)/(offset3-offset2));
                        if (sigmaX >= 0.6) {
                            //Log.d("TILTSHIFT_JAVA_THREAD", "Threshold Encounterd at:"+ i+",Sigma"+sigmaX);
                            //Determine the Kernel radius and assign indices using the increasing sigma values
                            int[] k = calK(sigmaX);                        //Calculating radius vector
                            //Determine the gaussian weights using the kernel indices and the sigma value
                            double[] kernelMatrix = GaussianWeightCal(k, sigmaX);  //Calculate the Gaussian Matrix

                            // Convolution for all the channels with the gaussian weight matrix
                            for (int j = 0; j < imgWidth; j++) {
                                //Compute the result of convolution with Gaussian kernel for each R,G,B,A color channel
                                int pBB = pVector("BB", j, i, imgWidth, k, kernelMatrix, pixels);
                                int pRR = pVector("RR", j, i, imgWidth, k, kernelMatrix, pixels);
                                int pGG = pVector("GG", j, i, imgWidth, k, kernelMatrix, pixels);
                                int pAA = pVector("AA", j, i, imgWidth, k, kernelMatrix, pixels);
                                // Copy the new pixel values to pixelsOut
                                pixelsOut[i * imgWidth + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                            }
                        }
                        //
                        // If sigma is below the  threshold value the pixels are left untouched
                        //
                        else {
                            //Log.d("TILTSHIFT_JAVA_THREAD", "coping from:"+ i+",Sigma"+sigmaX);
                            for(int z = i; z>offset2 ; z--) {
                                for (int j = 0; j < imgWidth; j++) {
                                    pixelsOut[i * imgWidth + j] = pixels[i * imgWidth + j];
                                }
                            }
                            break;
                        }
                    }
                }
                //Log.d("TILTSHIFT_JAVA_THREAD", "Running " + threadName + "Length of pixelsOut is" + pixelsOut.length);
            }
            catch (Exception e)
            {
                // Throwing an exception
                System.out.println ("Exception is caught"+ threadName +e.toString());
            }
        }
    }

    public static Bitmap tiltshift_java(Bitmap input, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3){
        long javaStart = System.currentTimeMillis(); //start timer
        // Determine the width and height of the image
        int imgHeight = input.getHeight();
        int imgWidth = input.getWidth();
        //Create a bitmap object of same with and height as the input
        Bitmap outBmp = Bitmap.createBitmap(imgWidth, imgHeight, Bitmap.Config.ARGB_8888);
        //cannot write to input Bitmap, since it may be immutable
        //if you try, you may get a java.lang.IllegalStateException
        int[] pixels = new int[imgHeight*imgWidth];
        input.getPixels(pixels,0,imgWidth,0,0,imgWidth,imgHeight);
        //During testing hard coding the values
//        a0=64;
//        a1=128;
//        a2=192;
//        a3=256;
//        sigma_far = (float) 4.5;
//        sigma_near = 3;
        Log.d("TILTSHIFT_JAVA","Started:"+imgWidth+","+imgHeight+","+pixels.length+"a0:"+a0+", a1:"+a1+", a2:"+a2+", a3:"+a3);
        // Main Class
        // Create instances to compute the new pixel values
        // Each thread instances operate on different regions of the original image
        // The image is sectioned into different regions by using the values set by the slider on the UI
        Sigma_far_near SigmaFarSolid = new Sigma_far_near("Thread-sigma-far-solid",sigma_far,a0,a1,a2,a3,pixels,imgHeight,imgWidth);
        Sigma_far_near SigmaNearSolid = new Sigma_far_near("Thread-sigma-near-solid",sigma_near,a0,a1,a2,a3,pixels,imgHeight,imgWidth);
        Sigma_far_near SigmaBlur = new Sigma_far_near("Thread-sigma-no-blur",sigma_near,a0,a1,a2,a3,pixels,imgHeight,imgWidth);
        Sigma_far_near SigmaFarGradual = new Sigma_far_near("Thread-sigma-far-gradual",sigma_far,a0,a1,a2,a3,pixels,imgHeight,imgWidth);
        Sigma_far_near SigmaNearGradual = new Sigma_far_near("Thread-sigma-near-gradual",sigma_near,a0,a1,a2,a3,pixels,imgHeight,imgWidth);

        //Initiate the threads to begin computation
        SigmaFarSolid.start();
        SigmaNearSolid.start();
        SigmaBlur.start();
        SigmaFarGradual.start();
        SigmaNearGradual.start();

        try {
            SigmaFarSolid.join();
            SigmaNearSolid.join();
            SigmaBlur.join();
            SigmaFarGradual.join();
            SigmaNearGradual.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        outBmp.setPixels(SigmaNearGradual.pixelsOut,0,imgWidth,0,0,imgWidth,imgHeight);
        javaElapsedTime = System.currentTimeMillis() - javaStart;   //elapsed time calculation
        Log.d("TILTSHIFT_JAVA","Time Elapsed:"+javaElapsedTime);
        return outBmp;
    }
    // Cpp implementation
    public static Bitmap tiltshift_cpp(Bitmap input, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3){
//        buf.clear();
//        a0=64;
//        a1=128;
//        a2=192;
//        a3=256;
//        sigma_far = (float) 4.5;
//        sigma_near = 3;
        // Initialize the timer to record time elapsed
        long cppStart = System.currentTimeMillis();
        //Create a bitmap object of same with and height as the input
        Bitmap outBmp = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);
        int[] pixels = new int[input.getHeight()*input.getWidth()];
        int[] pixelsOut = new int[input.getHeight()*input.getWidth()];
        //Load the pixel values into pixels bitmap object
        input.getPixels(pixels,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        //Call the function and pass variables needed for the C++ native implementation
        tiltshiftcppnative(pixels,pixelsOut,input.getWidth(),input.getHeight(),sigma_far,sigma_near,a0,a1,a2,a3);
        //Load the new computd pixel values into the bitmap object
        outBmp.setPixels(pixelsOut,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        javaElapsedTime = System.currentTimeMillis() - cppStart ;   //elapsed time calculation
        Log.d("TILTSHIFT_C++","Time Elapsed:"+javaElapsedTime);
        return outBmp;
    }

    //Neon implementation
    public static Bitmap tiltshift_neon(Bitmap input, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3){
        long neonStart = System.currentTimeMillis();
//        a0=70;
//        a1=128;
//        a2=192;
//        a3=196;
//        sigma_far = (float) 4.5;
//        sigma_near = 3;
        // Initialize the timer to record time elapsed
        Bitmap outBmp = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);
        int[] pixels = new int[input.getHeight()*input.getWidth()];
        int[] pixelsOut = new int[input.getHeight()*input.getWidth()];
        //Load the pixel values into pixels bitmap object
        input.getPixels(pixels,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        //Call the function and pass variables needed for the Neon native implementation
        tiltshiftneonnative(pixels,pixelsOut,input.getWidth(),input.getHeight(),sigma_far,sigma_near,a0,a1,a2,a3);
        //Load the new computed pixel values into the bitmap object
        outBmp.setPixels(pixelsOut,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        javaElapsedTime = System.currentTimeMillis() - neonStart ;   //elapsed time calculation
        Log.d("TILTSHIFT_NEON","Time Elapsed:"+javaElapsedTime);
        return outBmp;
    }

    // function to compute the size of the kernel as per the sigma value and populate the kernel with indices
    public static int[] calK(float sigma) {
        // setting the radius to 2 Sigma
        int r = (int) Math.ceil(2 * sigma);
        int kernel_size = 2 * r + 1;
        int[] k = new int[kernel_size];             // initializing Kernel vector
        int m = -r;  // set initail point to -r
        // loop over the entire kernel and initialize as per increasing decreasing index
        for (int i = 0; i < kernel_size; i++) {
            k[i] = m++;
        }
        return k;
    }
    // function to compute the gaussian weights using the indices assigned in the kernel variable as per the above function
    public static double[] GaussianWeightCal(int[] k, double sigma){
        int len = k.length;
        double[] weight = new double[len];
        for(int i=0; i<len; i++){
            //Determine the wights by plugging in values in pdf of gaussian distribution
            weight[i]=(Math.exp((-1*k[i]*k[i])/(2*sigma*sigma))/Math.sqrt(2*Math.PI*sigma*sigma));
        }
        return weight;
    }

    // Calculates the final weighted sum by convolution with the gaussian weights
    // Access each color channel and compute the guassian blur and append it to the pixels
    public static int pVector(String channel, int x, int y, int width, int[] k, double[] G, int[] pixel){
        double P=0;
        int len = k.length;
        for(int i=0; i<len; i++){
            // Calculate the highest pixel to be accessed and verify if it is within the image dimensions
            if(y * width + x + k[i] < pixel.length) {
                //Computed the new value by calling the qVector function
                P = P + G[i] * qVector(channel, x + k[i], y, width, k, G, pixel);
            }
        }
        return (int)P ;
    }
    // Function to calculate the shifted pixel arrays  and perform 1D-convolution
    public static double qVector(String channel, int x, int y, int width, int[] k, double[] G, int[] pixels){

        double q=0;
        int p;
        int len = k.length;
        //The following switch-case based on the color AA RR BB GG
        switch(channel){
            case "BB": for(int i=0; i<len; i++){
                //zero padding for edges
                if((y+k[i])<0 || x<0 || (y+k[i])*width+x >= pixels.length){
                    //checking the conditions for the edge cases
                    p=0;
                }
                else{
                    // Assign the pixel values
                    p=pixels[(y+k[i])*width+x];
                }
                // Isolate the blue pixel value from the original pixel value
                int BB = p & 0xff;
                q+=BB*G[i];
            }
                return q;

            case "GG": for(int i=0; i<len; i++){
                if((y+k[i])<0 || x<0 || (y+k[i])*width+x>= pixels.length){
                    //checking the conditions for the edge cases
                    p=0;
                }
                else{
                    // Assign the pixel values
                    p=pixels[(y+k[i])*width+x];
                }
                // make a 8 bit right shift
                // Isolate the green pixel value from the original pixel value
                int GG = (p>>8) & 0xff;
                q+=GG*G[i];
            }
                return q;

            case "RR": for(int i=0; i<len; i++){
                if((y+k[i])<0 || x<0 || (y+k[i])*width+x>= pixels.length){
                    //checking the conditions for the edge cases
                    p=0;
                }
                else{
                    // Assign the pixel values
                    p=pixels[(y+k[i])*width+x];
                }
                // make a 16 bit right shift
                // Isolate the red pixel value from the original pixel value
                int RR = (p>>16) &0xff;
                q+=RR*G[i];
            }
                return q;

            case "AA": for(int i=0; i<len; i++){
                if((y+k[i])<0 || x<0 || (y+k[i])*width+x>= pixels.length){
                    //checking the conditions for the edge cases
                    p=0;
                }
                else {
                    // Assign the pixel values
                    p=pixels[(y+k[i])*width+x];
                }
                // make a 16 bit right shift
                // Isolate the red pixel value from the original pixel value
                int AA = (p>>24) &0xff;
                q+=AA*G[i];
            }
                return q;
        }
        return q;
    }
    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public static native int tiltshiftcppnative(int[] inputPixels, int[] outputPixels, int width, int height, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3);
    public static native int tiltshiftneonnative(int[] inputPixels, int[] outputPixels, int width, int height, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3);
}