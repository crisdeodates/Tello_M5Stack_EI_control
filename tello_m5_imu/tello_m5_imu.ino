/* Edge Impulse Arduino examples
 * Copyright (c) 2021 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#define M5STACK_MPU6886 

/* Includes ---------------------------------------------------------------- */
#include <M5Stack.h>
#include <Tello.h>
#include <drone_gesture_control_accel_inference.h>

/* Constant defines -------------------------------------------------------- */
#define CONVERT_G_TO_MS2    9.80665f

/* Private variables ------------------------------------------------------- */
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal

//Tello variables
const char * networkName = "TELLO-ABCCB9";
const char * networkPswd = "";
boolean connected = false;
int IsOnAir = -1;
Tello tello;

boolean do_control = false;

/**
* @brief      Arduino setup function
*/
void setup()
{    
    // Initialize the M5Stack object
    M5.begin();
    /*
      Power chip connected to gpio21, gpio22, I2C device
      Set battery charging voltage and current
      If used battery, please call this function in your project
    */
    M5.Power.begin();
      
    M5.IMU.Init();

    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setTextColor(GREEN , BLACK);
    M5.Lcd.setTextSize(2);
    
    // put your setup code here, to run once:
    Serial.begin(115200);delay(2000);
    Serial.println("Drone Gesture Control\nDesigned by Team Deodates\nProgrammers: Cris & Jiss\n--> Inference by EdgeImpulse\n\n");
    M5.Lcd.setCursor(0, 20);
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.print("Drone Gesture Control\n\n");
    M5.Lcd.print("Designed by Team Deodates\n\n");
    M5.Lcd.print("Programmers: Cris & Jiss\n\n");
    M5.Lcd.print("--> Inference by \n\t\tEdgeImpulse\n");
    delay(2000);
    
    //Connect to the WiFi network
    connectToWiFi(networkName, networkPswd);

    if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 3) {
        ei_printf("ERR: EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME should be equal to 3 (the 3 sensor axes)\n");
        return;
    }
}

/**
* @brief      Printf function uses vsnprintf and output using Arduino Serial
*
* @param[in]  format     Variable argument list
*/
void ei_printf(const char *format, ...) {
   static char print_buf[1024] = { 0 };

   va_list args;
   va_start(args, format);
   int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
   va_end(args);

   if (r > 0) {
       Serial.write(print_buf);
   }
}

/**
* @brief      Get data and run inferencing
*
* @param[in]  debug  Get debug info if true
*/
void loop()
{
    //Manual
    String result_label = "";
    float result_pred = 0;

    M5.update();
    if(M5.BtnA.wasPressed()) do_control = true;
    else if(M5.BtnB.wasPressed()) do_control = false;
    
    ei_printf("\nStarting inferencing in 0.5 seconds...\n");

    delay(500);

    ei_printf("Sampling...\n");

    // Allocate a buffer here for the values we'll read from the IMU
    float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
        // Determine the next tick (and then sleep later)
        uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

        //IMU.readAcceleration(buffer[ix], buffer[ix + 1], buffer[ix + 2]);
        M5.IMU.getAccelData(&buffer[ix], &buffer[ix + 1], &buffer[ix + 2]);

        buffer[ix + 0] *= CONVERT_G_TO_MS2;
        buffer[ix + 1] *= CONVERT_G_TO_MS2;
        buffer[ix + 2] *= CONVERT_G_TO_MS2;

        delayMicroseconds(next_tick - micros());
    }

    // Turn the raw buffer in a signal which we can the classify
    signal_t signal;
    int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    if (err != 0) {
        ei_printf("Failed to create signal from buffer (%d)\n", err);
        return;
    }

    // Run the classifier
    ei_impulse_result_t result = { 0 };

    err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        return;
    }

    // print the predictions
    ei_printf("Predictions ");
    ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
        result.timing.dsp, result.timing.classification, result.timing.anomaly);
    ei_printf(": \n");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);

        // Manual
        if((float)result.classification[ix].value > result_pred)
        {          
          result_pred = (float)result.classification[ix].value;
          result_label = String(result.classification[ix].label);
          ei_printf(" Outcome -->   %s: %.5f\n", result_label, result_pred);
        }
    }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif

    if(connected && do_control) tello_action(result_label);

    // Manual
    M5.Lcd.setCursor(0, 20);
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.print("Prediction \r\n");
    M5.Lcd.printf(" Label: %s \r\n", result_label);
    M5.Lcd.printf(" Score: %2.5f ", result_pred);
    M5.Lcd.print("\n\nInferencing in 0.5s... \r\n");
    M5.Lcd.printf("\n\nControl = %1.0f \r\n", (float)do_control);
    Serial.println("Control = " + String(do_control));      
}

void connectToWiFi(const char * ssid, const char * pwd) 
{
  Serial.println("Connecting to WiFi network: " + String(ssid));
  M5.Lcd.setCursor(0, 20);
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.printf("Wifi Connecting to %s \r\n", String(ssid));

  // delete old config
  WiFi.disconnect(true);
  //register event handler
  WiFi.onEvent(WiFiEvent);

  //Initiate connection
  WiFi.begin(ssid, pwd);

  Serial.println("Waiting for WIFI connection...");
  M5.Lcd.print("Waiting for WIFI connection... \r\n");  
}

//wifi event handler
void WiFiEvent(WiFiEvent_t event) 
{
  switch (event) 
  {
    case SYSTEM_EVENT_STA_GOT_IP:
      //When connected set
      Serial.print("WiFi connected! IP address: ");
      Serial.println(WiFi.localIP());      
      M5.Lcd.printf("\n\nWifi Connected to %s \r\n", String(WiFi.localIP()));  
      //initialise Tello after we are connected
      tello.init();
      connected = true;
      break;
      
    case SYSTEM_EVENT_STA_DISCONNECTED:
      Serial.println("WiFi lost connection");      
      M5.Lcd.print("\n\nWiFi lost connection \r\n");  
      connected = false;
      break;
  }
}

void tello_action(String &result_label)
{
    if(result_label == "takeoff" or result_label == "land") 
    {
      if(IsOnAir < 0) tello.takeoff();
      else if(IsOnAir > 0) tello.land();
      IsOnAir = -1 * IsOnAir;
    }
    else if(result_label == "forward") tello.forward(40); //cm
    else if(result_label == "back") tello.back(40); //cm
    else if(result_label == "left") tello.left(40); //cm
    else if(result_label == "right") tello.right(40); //cm 
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_ACCELEROMETER
#error "Invalid model for current sensor"
#endif
