// --- Includes --- //
#include <stdlib.h> // ?needed?
#include <string.h>
#include <math.h> // ?
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include "PerspectiveTracker.h"
#include "png.h"
#include "walls.h"

#include "opencv2/video/video.hpp"

// --- Definitions --- //
#define SCREEN_WIDTH        0.333 // Have at aspect ratio of screen
#define SCREEN_HEIGHT       0.207


#define FRAME_WIDTH         480
#define FRAME_HEIGHT        320
#define FRAMES_BEFORE_NEW   20

#define EYE_MAX_SPEED       0.05 // ACTUALLY IMPLEMENT TIMEING RATHER THAN LEAVING IT UP TO DISPLAY TIMES

#define KEY_ESC             27
#define KEY_ENTER           13

#define RAD2DEG             (180.0/M_PI)

// HAVE A CONSTANT FOR ASPECT
//#define WALL_WIDTH          1.44 // Have at aspect ratio of screen
//#define WALL_HEIGHT         0.9
#define WALL_WIDTH          SCREEN_WIDTH
#define WALL_HEIGHT         SCREEN_HEIGHT
#define WALL_FRONT_DEPTH    0.0
#define WALL_DEPTH          0.2
#define WALL_RESOLUTION     10

#define BLOCK_SIZE          10 // larger is smaller
#define BLOCK_GAP           0.75 // inverse gap size (1 = no gap, 0.5 = half block)
#define BLOCK_STILL         100 // percentage of still blocks
#define BLOCK_NUM_X         (144/BLOCK_SIZE)
#define BLOCK_NUM_Y         (90/BLOCK_SIZE)
#define BLOCK_WIDTH         (WALL_WIDTH/BLOCK_NUM_X)
#define BLOCK_HEIGHT        (WALL_HEIGHT/BLOCK_NUM_Y)
#define BLOCK_MAX_DEPTH     1.5
#define BLOCK_MIN_DEPTH     0.1
#define BLOCK_MAX_RATE      0.01
#define BLOCK_NUM_COLOURS   6
#define BLOCK_MENISCUS      0.5

#define AXIS_LENGTH         20

enum filterTypes {FILTER_AVERAGE, FILTER_ALPHA_BETA, FILTER_NONE, FILTER_KALMAN, FILTER_COUNT};

// --- Macros --- //
//#define sign(x)             (signbit(x) ? 1 : -1)


// --- Structures --- //
typedef struct {
    GLfloat x;
    GLfloat y;
    GLfloat z;
} point3f;

typedef struct {
    point3f start;
    point3f end;
} lineSeg3f;

// --- Adjustable Parameters --- //

int blockColours[BLOCK_NUM_COLOURS];
GLfloat blockDepths[BLOCK_NUM_X][BLOCK_NUM_Y];
GLfloat blockRates[BLOCK_NUM_X][BLOCK_NUM_Y];
// GLfloat blockAccelerations[BLOCK_NUM_X][BLOCK_NUM_Y];
int blockColourNums[BLOCK_NUM_X][BLOCK_NUM_Y];


int window, modelWindow;
String resourcesDirectory = "/Users/dan_vogelnest/Documents/Projects/P3DI/Resources/";
//String resourcesDirectory = "/Users/Satveer/Dropbox/Major Assignment/Satveer/P3DI/Resources/";
String haarcascadeDirectory = resourcesDirectory + "haarcascades/";
String imageName = "landscape.png";

point3f eye, oldEye;
PerspectiveTracker perspectiveTracker;
CvCapture* camera;
const char* window_name = "Capture - Face detection";
vector<double> model2CameraVector;
static GLuint texture;
bool fullscreen = false;
bool modelFullscreen = false;
bool pictureView = false;
int filterType = FILTER_AVERAGE;
GLfloat weight = 0.4;
point3f lastEye, eyeVelocity, lastEyeVelocity, eyeError;
float alpha = 0.85;
float beta = 2*(2-alpha) - 4*sqrt(1-alpha);//0.005;//
int lastTime = 0;

const float A[] = //{1, 1, 0, 1}; 
{
    1, 0, 1, 0, 0, 0,
    0, 1, 0, 1, 0, 0,
    0, 0, 1, 0, 1, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1
};

CvKalman* kalman;
CvMat* state = NULL;
CvMat* measurement;

enum colourNames {C_RED_D, C_RED_L, C_GREEN_D, C_GREEN_L, C_BLUE_D, C_BLUE_L, C_YELLOW, C_BROWN_L, C_BLACK, C_GREY, C_WHITE};

float colours[][3] =
{
    // Red
    {0.4, 0.1, 0.0},
    {0.8, 0.1, 0.0},
    
    // Green
    {0.2, 0.4, 0.1},
    {0.0, 0.5, 0.2},
    
    // Blue
    {0.1, 0.2, 0.3},
    {0.1, 0.5, 0.7},
    
    // Yellow
    {0.8, 0.8, 0.0},
    {0.6, 0.4, 0.2},
    
    // Grey
    {0.0, 0.0, 0.0},
    {0.5, 0.5, 0.5},
    {1.0, 1.0, 1.0}
};

void idle(void);
void display(void);
void displayModel(void);
void initOpenGLcontext(void);
void keyboard(unsigned char key, int x, int y);
void reshapeWindow(int w, int h);
void reshapeModelWindow(int w, int h);
void buildBlocks(void);
void drawString(const char* string, GLfloat size);
void drawWalls(void);
void drawMesh(void);
void drawBlocks(void);
void drawAxis(GLfloat resolution, GLfloat ticSize);
void drawGrid(GLfloat halfWidth, GLfloat halfHeight, GLfloat resolution);
GLuint loadTexture(const char* filename, int &width, int &height);