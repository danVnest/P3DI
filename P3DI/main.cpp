// --- P3DI --- //
// --- Perspective Three Dimensional Imaging --- //

#include "main.h"

// --- Main Function --- //
int main (int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); 
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(glutGet(GLUT_SCREEN_WIDTH)/2, glutGet(GLUT_SCREEN_HEIGHT)/2);
    glutIdleFunc(idle);
    
    // Model View
    modelWindow = glutCreateWindow("Model View");
    glutPositionWindow(glutGet(GLUT_SCREEN_WIDTH)/2, glutGet(GLUT_SCREEN_HEIGHT)/2);
    glutDisplayFunc(displayModel);
    glutReshapeFunc(reshapeModelWindow);
    glutKeyboardFunc(keyboard);
    initOpenGLcontext();
    
    // Main Window
    window = glutCreateWindow("P3DI");
    glutPositionWindow(0, glutGet(GLUT_SCREEN_HEIGHT)/2);
    glutSetCursor(GLUT_CURSOR_NONE);
    glutDisplayFunc(display);
    glutReshapeFunc(reshapeWindow);
    glutKeyboardFunc(keyboard);   
    initOpenGLcontext();
    
    int width, height;
    String filename = resourcesDirectory + imageName;
    texture = loadTexture(filename.c_str(), width, height);
    
    // 3D Space Configuration
    kalman =  cvCreateKalman(6, 3, 0);
    state = cvCreateMat(6, 1, CV_32FC1);
    cvZero(state);
    measurement = cvCreateMat(3, 1, CV_32FC1);
    cvZero(measurement);    
    
    memcpy(kalman->transition_matrix->data.fl, A, sizeof(A));
    cvSetIdentity(kalman->measurement_matrix,cvRealScalar(1));    
    cvSetIdentity(kalman->process_noise_cov, cvRealScalar(0.05));
    cvSetIdentity(kalman->measurement_noise_cov, cvRealScalar(0.1));
    cvSetIdentity(kalman->error_cov_post, cvRealScalar(0));
    
    //kalman->state_post->data.fl[2] = 1;
    
    eye.x = 0.0;
    eye.y = 0.0;
    eye.z = 1.0;
    
    model2CameraVector.resize(3);
    model2CameraVector[0] = 0;
    model2CameraVector[1] = 0; 
    model2CameraVector[2] = 0;
    
    buildBlocks();
    
    perspectiveTracker = PerspectiveTracker(CAMERA, FACE, FRAME_WIDTH, FRAME_HEIGHT, model2CameraVector, FRAMES_BEFORE_NEW, haarcascadeDirectory);

    glutMainLoop();
}

// --- Idle --- //
void idle(void)
{
    unsigned char key = waitKey(1);
    if (key != 0xFF) keyboard(key, 0, 0);
    
    vector<double> model2PersonVector = perspectiveTracker.GetPosition();
    point3f frameEye;
    frameEye.x = model2PersonVector[0];
    frameEye.y = model2PersonVector[1];
    frameEye.z = model2PersonVector[2];
    
    if (filterType == FILTER_NONE) eye = frameEye;
    else if (filterType == FILTER_AVERAGE) 
    {
        eye.x = (1 - weight)*eye.x + weight*frameEye.x;
        eye.y = (1 - weight)*eye.y + weight*frameEye.y;
        eye.z = (1 - weight)*eye.z + weight*frameEye.z;
    }
    else if (filterType == FILTER_ALPHA_BETA)
    {
        int time = glutGet(GLUT_ELAPSED_TIME);
        double timeChange = (time - lastTime)/1000.0;
        lastTime = time;
        eye.x = lastEye.x + lastEyeVelocity.x*timeChange;
        eye.y = lastEye.y + lastEyeVelocity.y*timeChange;
        eye.z = lastEye.z + lastEyeVelocity.z*timeChange;

        eyeVelocity.x = lastEyeVelocity.x;
        eyeVelocity.y = lastEyeVelocity.y;
        eyeVelocity.z = lastEyeVelocity.z;

        eyeError.x = frameEye.x - eye.x;
        eyeError.y = frameEye.y - eye.y;
        eyeError.z = frameEye.z - eye.z;

        eye.x += alpha*eyeError.x;
        eye.y += alpha*eyeError.y;
        eye.z += alpha*eyeError.z;
        
        eyeVelocity.x += beta*eyeError.x/timeChange;
        eyeVelocity.y += beta*eyeError.y/timeChange;
        eyeVelocity.z += beta*eyeError.z/timeChange;

        lastEye.x = eye.x;
        lastEye.y = eye.y;
        lastEye.z = eye.z;
        
        lastEyeVelocity.x = eyeVelocity.x;
        lastEyeVelocity.y = eyeVelocity.y;
        lastEyeVelocity.z = eyeVelocity.z;
    }
    else if (filterType == FILTER_KALMAN)
    {
        eye = frameEye;
        measurement->data.fl[0] = eye.x;
        measurement->data.fl[1] = eye.y;
        measurement->data.fl[2] = eye.z;
        cvKalmanCorrect(kalman, measurement);
        const CvMat* prediction = cvKalmanPredict(kalman, 0);
        //eye.x = prediction->data.fl[0];
        eye.y = prediction->data.fl[1];
        eye.z = prediction->data.fl[2];

        oldEye = eye;
    }
    glutSetWindow(window);
    glutPostRedisplay();
    glutSetWindow(modelWindow);
    glutPostRedisplay();
}



// --- Display Callback --- //
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(	0.01*(-WALL_WIDTH/2 - eye.x)/eye.z,
              0.01*(WALL_WIDTH/2 - eye.x)/eye.z,
              0.01*(-WALL_HEIGHT/2 - eye.y)/eye.z,
              0.01*(WALL_HEIGHT/2 - eye.y)/eye.z,
              0.01, 100.0);
    glTranslatef(-eye.x, -eye.y, -eye.z);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if (pictureView) 
    {
        glPushMatrix();
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glTranslatef(0.0, 0.0, -50.0);
        glScalef(20, 20, 20);
        glEnable(GL_TEXTURE_2D);
        glDisable(GL_LIGHTING);
        glColor3fv(colours[C_WHITE]);
        glBegin(GL_QUADS);
        glNormal3f(0.0, 0.0, 1.0);
        glTexCoord2d(0, 0); glVertex3f(-1.0, -1.0, 0.0);
        glTexCoord2d(0, 1); glVertex3f(-1.0, 1.0, 0.0);
        glTexCoord2d(1, 1); glVertex3f(1.0, 1.0, 0.0);
        glTexCoord2d(1, 0); glVertex3f(1.0, -1.0, 0.0);
        glEnd();
        glDisable(GL_TEXTURE_2D);
        glEnable(GL_LIGHTING);
        glPopAttrib();
        glPopMatrix();
        glFlush();
    }
    else
    {
        drawBlocks();
        drawWalls();
        drawMesh();
    }
    if (glGetError()) exit(EXIT_FAILURE);
    glutSwapBuffers();
}


// --- Display Model --- //
void displayModel(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity(); 

    GLsizei viewWidth = glutGet(GLUT_WINDOW_WIDTH);
    GLsizei viewHeight = glutGet(GLUT_WINDOW_HEIGHT)/3;
    
    // Front View
    glPushMatrix();
    glViewport(0, viewHeight, viewWidth, 2*viewHeight);
    glScalef(2.0/3.0, 1, 1);
    float scale = 0.333/SCREEN_WIDTH;
    glScalef(scale, scale, scale);
    gluLookAt(0.0, 0.0, 50,
              0.0, 0.0, 0.0,
              0.0, 1.0, 0.0);
    glPushMatrix();
    glTranslatef(0.0, 0.0, BLOCK_MAX_DEPTH-WALL_DEPTH);
    drawAxis(0.01, 0.005);
    glRotatef(-90.0, 0.0, 0.0, 1.0);
    drawAxis(0.01, 0.005);
    glPopMatrix();
    drawWalls();
    drawMesh();
    drawBlocks();
    glPushMatrix();
    glTranslatef(eye.x, eye.y, eye.z+BLOCK_MAX_DEPTH-WALL_DEPTH);
    glMaterialfv(GL_FRONT, GL_AMBIENT, colours[C_RED_L]);
    glMaterialfv(GL_FRONT, GL_EMISSION, colours[C_RED_L]);
    glutSolidSphere(0.02, 10, 10);
    glPopMatrix();
    glPushMatrix();
    glMaterialfv(GL_FRONT, GL_AMBIENT, colours[C_BLUE_L]);
    glMaterialfv(GL_FRONT, GL_EMISSION, colours[C_BLUE_L]);
    glPopMatrix();
    glPopMatrix();
    
    // Side View
    glPushMatrix();
    glViewport(0, 0, viewWidth, viewHeight);
    glScalef(1.0/3.0, 1, 1);
    glScalef(scale, scale, scale);
    glTranslatef(-0.6, 0.0, 0.0);
    glScalef(0.75, 0.75, 0.75);
    gluLookAt(-50, 0.0, 0.0,
              0.0, 0.0, 0.0,
              0.0, 1.0, 0.0);
    GLfloat lightPosition[] = {-WALL_WIDTH/2, 0, 10.0, 1.0};
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    drawWalls();
    glPushMatrix();
    glScaled(1.02, 1.02, 1.02);
    drawMesh();
    glPopMatrix();
    drawBlocks();
    glTranslatef(-1.0, 0.0, 0.0);
    glPushMatrix();
    glRotatef(-90.0, 0.0, 1.0, 0.0);
    drawAxis(0.1, 0.01);
    glRotatef(-90.0, 0.0, 0.0, 1.0);
    drawAxis(0.1, 0.01);
    glPopMatrix();
    glPushMatrix();
    glTranslatef(eye.x-1, eye.y, eye.z);
    glMaterialfv(GL_FRONT, GL_AMBIENT, colours[C_RED_L]);
    glMaterialfv(GL_FRONT, GL_EMISSION, colours[C_RED_L]);
    glutSolidSphere(0.02, 10, 10);
    glPopMatrix();
    glPopMatrix();
    
    // Text
    glPushMatrix();
    glViewport(0, 0, viewWidth, 3*viewHeight);
    gluLookAt(0.0, 0.0, 50,
              0.0, 0.0, 0.0,
              0.0, 1.0, 0.0);
    
    glPushMatrix();
    glMaterialfv(GL_FRONT, GL_AMBIENT, colours[C_GREEN_L]);
    glMaterialfv(GL_FRONT, GL_EMISSION, colours[C_GREEN_L]);
    glTranslatef(0.19, 0.18, 0.0);
    if (filterType == FILTER_NONE) drawString("Filter: None", 0.01);
    else if (filterType == FILTER_AVERAGE) 
    {
        glTranslatef(-0.05, 0.0, 0.0);
        char string[0xFF];
        sprintf(string, "Filter: LPF, w=%0.1f", weight);
        drawString(string, 0.01);
    }
    else if (filterType == FILTER_ALPHA_BETA)
    {
        glTranslatef(-0.05, 0.0, 0.0);
        char string[0xFF];
        sprintf(string, "Filter: å=%0.1f ß=%f", alpha, beta);
        drawString(string, 0.01);
    }
    else if (filterType == FILTER_KALMAN) drawString("Filter: Kalman", 0.01);
    glPopMatrix();
    
    glMaterialfv(GL_FRONT, GL_AMBIENT, colours[C_BLUE_L]);
    glMaterialfv(GL_FRONT, GL_EMISSION, colours[C_BLUE_L]);
    
    // Front View
    glPushMatrix();
    glPushMatrix();
    glTranslatef(-0.3, 0.16, 0);
    drawString("Front View", 0.015);
    glPopMatrix();
    glPushMatrix();
    glTranslatef(0.27, 0.045, 0);
    drawString("x (cm)", 0.0075);
    glPopMatrix();
    glPushMatrix();
    glTranslatef(-0.015, 0.18, 0);
    drawString("y", 0.0075);
    glPopMatrix();
    glPopMatrix();
    
    // Side View
    glPushMatrix();
    glPushMatrix();
    glTranslatef(-0.3, -0.075, 0);
    drawString("Side View", 0.015);    
    glPopMatrix();
    glPushMatrix();
    glTranslatef(0.27, -0.155, 0);
    drawString("z (dm)", 0.0075);
    glPopMatrix();
    glPushMatrix();
    glTranslatef(-0.145, -0.075, 0);
    drawString("y", 0.0075);
    glPopMatrix();
    glPopMatrix();
    glPopMatrix();
    
    if (glGetError()) exit(EXIT_FAILURE);
    glutSwapBuffers();
}

// --- Respond to Key Press --- //
void keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case KEY_ESC:
            exit(EXIT_SUCCESS);
            break;
            
        case KEY_ENTER:
            glutSetWindow(window);
            if (!fullscreen) glutFullScreen();
            else 
            {
                glutReshapeWindow(glutGet(GLUT_SCREEN_WIDTH)/2, glutGet(GLUT_SCREEN_HEIGHT)/2);
                glutPositionWindow(0, glutGet(GLUT_SCREEN_HEIGHT)/2);
            }
            fullscreen = !fullscreen;
            if (modelFullscreen) 
            {
                glutSetWindow(modelWindow);
                glutReshapeWindow(glutGet(GLUT_SCREEN_WIDTH)/2, glutGet(GLUT_SCREEN_HEIGHT)/2);
                glutPositionWindow(glutGet(GLUT_SCREEN_WIDTH)/2, glutGet(GLUT_SCREEN_HEIGHT)/2);
                modelFullscreen = !modelFullscreen;
            }
            break;
            
        case 'v':
            glutSetWindow(modelWindow);
            if (!modelFullscreen) glutFullScreen();
            else 
            {
                glutReshapeWindow(glutGet(GLUT_SCREEN_WIDTH)/2, glutGet(GLUT_SCREEN_HEIGHT)/2);
                glutPositionWindow(glutGet(GLUT_SCREEN_WIDTH)/2, glutGet(GLUT_SCREEN_HEIGHT)/2);   
            }
            modelFullscreen = !modelFullscreen;
            if (fullscreen) 
            {
                glutSetWindow(window);
                glutReshapeWindow(glutGet(GLUT_SCREEN_WIDTH)/2, glutGet(GLUT_SCREEN_HEIGHT)/2);
                glutPositionWindow(0, glutGet(GLUT_SCREEN_HEIGHT)/2);
                fullscreen = !fullscreen;
            }
            break;
            
        case 'p':
            pictureView = !pictureView;
            if (pictureView) 
            {
                glDisable(GL_LIGHTING);
            }
            else glEnable(GL_LIGHTING);
            break;
            
        case 'f':
            filterType++;
            if (filterType == FILTER_COUNT) filterType = 0;
            break;
        
        case 't':
            perspectiveTracker.ToggleTrackingMode();
            break;
            
        case 'c':
            perspectiveTracker.ToggleWindowSize();
            break;
            
        case '=': // +
            if (filterType != FILTER_AVERAGE) break;
            weight += 0.1;
            if (weight > 1) weight = 1;
            break;
            
        case '-':
            if (filterType != FILTER_AVERAGE) break;
            weight -= 0.1;
            if (weight < 0) weight = 0;
            break;
            
        default:
            break;
    }
}


// --- Respond to Window Reshape --- //
void reshapeWindow(int w, int h)
{
    glViewport(0, 0, (GLsizei) w, (GLsizei) h); 
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //gluPerspective(50, 144/90, 0.01, 1000.0);
    //adjustPerspective(); // If screen changes size, perspective does too... consider keeping objects same size, but the view... therefore needs a different function...
    glMatrixMode(GL_MODELVIEW);
}


void reshapeModelWindow(int w, int h)
{
    glViewport(0, 0, (GLsizei) w, (GLsizei) h); 
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    GLfloat aspect = (GLfloat)w/h;
    glOrtho(-aspect/5, aspect/5, -0.2, 0.2, 0.01, 100.0);
    glMatrixMode(GL_MODELVIEW);
}


// --- Initialise OpenGL --- //
void initOpenGLcontext(void)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    
    GLfloat lightColour[] = {0.2, 0.3, 0.4, 1.0};
    GLfloat lightScatter[] = {0.1, 0.1, 0.1, 1.0};
    GLfloat lightPosition[] = {WALL_WIDTH/2, WALL_HEIGHT/2, 1.0, 1.0};
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightColour);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightScatter);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightScatter);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    
    glShadeModel(GL_SMOOTH);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
}


// --- Draw a String --- //
void drawString(const char* string, GLfloat size)
{
    glPushMatrix();
    GLfloat height = glutStrokeWidth(GLUT_STROKE_ROMAN, 'X');
    GLfloat scale = size/height;
    glScalef(scale, scale, 1.0);
    glLineWidth(2);
    char* p = (char*) string;
    while (*p != '\0') glutStrokeCharacter(GLUT_STROKE_ROMAN, *p++);
    glLineWidth(1);
    glPopMatrix();
}

// --- Build Block Model --- //
void buildBlocks(void)
{
    // Configure Blocks
    blockColours[0] = C_RED_D;
    blockColours[1] = C_GREEN_D;
    blockColours[2] = C_GREEN_L;
    blockColours[3] = C_BLUE_D;
    blockColours[4] = C_BLUE_L;
    blockColours[5] = C_BROWN_L;
    
    for (int i = 0; i < BLOCK_NUM_X; i++)
    {
        for (int j = 0; j < BLOCK_NUM_Y; j++)
        {
            blockDepths[i][j] = BLOCK_MAX_DEPTH*WALL_DEPTH*(BLOCK_MIN_DEPTH + (1-BLOCK_MIN_DEPTH)*((float)rand())/RAND_MAX);
            if (rand() % 100 > BLOCK_STILL)
                blockRates[i][j] = BLOCK_MAX_RATE*(1-2*((float)rand())/RAND_MAX);
            else 
                blockRates[i][j] = 0;
            blockColourNums[i][j] = rand() % BLOCK_NUM_COLOURS;
        }
    }
}


// --- Draw Blocks --- //
void drawBlocks(void)
{
    glPushMatrix();
    GLfloat blockSpecular[] = {0.0, 0.0, 0.0, 1.0};
    GLfloat blockShininess[] = {0.0};
    GLfloat blockEmission[] = {0.4, 0.3, 0.4, 1.0}; // CHANGE FOR AWESOMENESS
    glMaterialfv(GL_FRONT, GL_SPECULAR, blockSpecular);
    glMaterialfv(GL_FRONT, GL_SHININESS, blockShininess);
    glMaterialfv(GL_FRONT, GL_EMISSION, blockEmission);
    glScalef(BLOCK_MENISCUS, BLOCK_MENISCUS, 1.0);
    glEnable(GL_COLOR_MATERIAL);
    for (int i = 0; i < BLOCK_NUM_X; i++)
    {
        for (int j = 0; j < BLOCK_NUM_Y; j++)
        {
            glPushMatrix();
            GLfloat blockDepth = blockDepths[i][j];
            glTranslatef(-WALL_WIDTH/2 + (2*i+1)*BLOCK_WIDTH/2, -WALL_HEIGHT/2 + (2*j+1)*BLOCK_HEIGHT/2, blockDepth/2 - WALL_DEPTH);
            glScalef(BLOCK_WIDTH, BLOCK_HEIGHT, blockDepth/BLOCK_GAP);
            glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
            glColor3fv(colours[blockColours[blockColourNums[i][j]]]);
            glutSolidCube(BLOCK_GAP);
            glPopMatrix();
            GLfloat blockRate = blockRates[i][j];
            if ((blockDepth + blockRate > BLOCK_MAX_DEPTH*WALL_DEPTH) || (blockDepth + blockRate <= BLOCK_MIN_DEPTH*WALL_DEPTH))
                blockRates[i][j] = -blockRate; //maybe sin acceleration
            blockDepths[i][j] = blockDepth + blockRates[i][j];
        }
    }
    glDisable(GL_COLOR_MATERIAL);
    glPopMatrix();
}


// --- Draw Walls --- //
void drawWalls(void)
{
    glPushMatrix();
    glScalef(WALL_WIDTH/2, WALL_HEIGHT/2, WALL_DEPTH);
    glMaterialfv(GL_FRONT, GL_AMBIENT, colours[C_BLUE_D]);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, colours[C_BLUE_L]);
    glMaterialfv(GL_FRONT, GL_SPECULAR, colours[C_BLUE_L]);
    glMaterialf(GL_FRONT, GL_SHININESS, 100);
    glMaterialfv(GL_FRONT, GL_EMISSION, colours[C_BLACK]);
    glVertexPointer(3, GL_FLOAT, 0, wallsVertices);
    glNormalPointer(GL_FLOAT, 0, wallsNormals);
    glDrawArrays(GL_TRIANGLES, 0, 24);

    // Back Wall
    glBegin(GL_POLYGON);
    glVertex3f(-1, 1, -1);
    glVertex3f(1, 1, -1);
    glVertex3f(1, -1, -1);
    glVertex3f(-1, -1, -1);
    glEnd();
    glPopMatrix();
}

void drawMesh(void)
{
    glPushMatrix();
    glScaled(0.99, 0.99, 0.99);
    glPushMatrix();
    glTranslatef(0.0, 0.0, -WALL_DEPTH);
    drawGrid(WALL_WIDTH/2, WALL_HEIGHT/2, WALL_RESOLUTION);
    glPopMatrix();
    glPushMatrix();
    glTranslatef(0.0, WALL_HEIGHT/2, -WALL_DEPTH/2);
    glRotatef(90, 1.0, 0.0, 0.0);
    drawGrid(WALL_WIDTH/2, WALL_DEPTH/2, WALL_RESOLUTION);
    glTranslatef(0.0, 0.0, WALL_HEIGHT);
    drawGrid(WALL_WIDTH/2, WALL_DEPTH/2, WALL_RESOLUTION);
    glPopMatrix();
    glPushMatrix();
    glTranslatef(-WALL_WIDTH/2, 0.0, -WALL_DEPTH/2);
    glRotatef(90, 0.0, 1.0, 0.0);
    drawGrid(WALL_DEPTH/2, WALL_HEIGHT/2, WALL_RESOLUTION);
    glTranslatef(0.0, 0.0, WALL_WIDTH);
    drawGrid(WALL_DEPTH/2, WALL_HEIGHT/2, WALL_RESOLUTION);
    glPopMatrix();
    glPopMatrix();  
}

void drawAxis(GLfloat resolution, GLfloat ticSize)
{
    glPushMatrix();
    glMaterialfv(GL_FRONT, GL_AMBIENT, colours[C_YELLOW]);
    glMaterialfv(GL_FRONT, GL_EMISSION, colours[C_YELLOW]);
    glBegin(GL_LINES);
    glVertex3f(-AXIS_LENGTH/2, 0.0, 0.0);
    glVertex3f(AXIS_LENGTH/2, 0.0, 0.0);
    int i = 0;
    for (GLfloat x = -AXIS_LENGTH/2; x <= AXIS_LENGTH/2; x += resolution)
    {
        glVertex3f(x, 0.0, 0.0);
        if (i++ % 10 == 0) glVertex3f(x, 2*ticSize, 0.0);
        else glVertex3f(x, ticSize, 0.0);
    }
    glEnd();
    glPopMatrix();
}

void drawGrid(GLfloat halfWidth, GLfloat halfHeight, GLfloat resolution)
{
    glPushMatrix();
    glMaterialfv(GL_FRONT, GL_AMBIENT, colours[C_BLUE_D]);
    glMaterialfv(GL_FRONT, GL_EMISSION, colours[C_BLUE_D]);
    GLfloat xIncrement = 2*halfWidth/resolution;
    GLfloat yIncrement = 2*halfHeight/resolution;
    glBegin(GL_LINES);
    for (GLfloat x = -halfWidth; x <= halfWidth; x += xIncrement) {
        glVertex3f(x, -halfHeight, 0.0);
        glVertex3f(x, halfHeight, 0.0);
    }
    for (GLfloat y = -halfHeight; y <= halfHeight; y += yIncrement)
    {
        glVertex3f(-halfWidth, y, 0.0);
        glVertex3f(halfWidth, y, 0.0);
    }
    glEnd();
    glPopMatrix();
}

#define TEXTURE_LOAD_ERROR 0
GLuint loadTexture(const char* filename, int &width, int &height) 
{
    //header for testing if it is a png
    png_byte header[8];
    
    //open file as binary
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return TEXTURE_LOAD_ERROR;
    }
    
    //read the header
    fread(header, 1, 8, fp);
    
    //test if png
    int is_png = !png_sig_cmp(header, 0, 8);
    if (!is_png) {
        fclose(fp);
        return TEXTURE_LOAD_ERROR;
    }
    
    //create png struct
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL,
                                                 NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return (TEXTURE_LOAD_ERROR);
    }
    
    //create png info struct
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, (png_infopp) NULL, (png_infopp) NULL);
        fclose(fp);
        return (TEXTURE_LOAD_ERROR);
    }
    
    //create png info struct
    png_infop end_info = png_create_info_struct(png_ptr);
    if (!end_info) {
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) NULL);
        fclose(fp);
        return (TEXTURE_LOAD_ERROR);
    }
    
    //png error stuff, not sure libpng man suggests this.
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        fclose(fp);
        return (TEXTURE_LOAD_ERROR);
    }
    
    //init png reading
    png_init_io(png_ptr, fp);
    
    //let libpng know you already read the first 8 bytes
    png_set_sig_bytes(png_ptr, 8);
    
    // read all the info up to the image data
    png_read_info(png_ptr, info_ptr);
    
    //variables to pass to get info
    int bit_depth, color_type;
    png_uint_32 twidth, theight;
    
    // get info about png
    png_get_IHDR(png_ptr, info_ptr, &twidth, &theight, &bit_depth, &color_type,
                 NULL, NULL, NULL);
    
    //update width and height based on png info
    width = twidth;
    height = theight;
    
    // Update the png info struct.
    png_read_update_info(png_ptr, info_ptr);
    
    // Row size in bytes.
    png_size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    
    // Allocate the image_data as a big block, to be given to opengl
    png_byte *image_data = new png_byte[rowbytes * height];
    if (!image_data) {
        //clean up memory and close stuff
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        fclose(fp);
        return TEXTURE_LOAD_ERROR;
    }
    
    //row_pointers is for pointing to image_data for reading the png with libpng
    png_bytep *row_pointers = new png_bytep[height];
    if (!row_pointers) {
        //clean up memory and close stuff
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        delete[] image_data;
        fclose(fp);
        return TEXTURE_LOAD_ERROR;
    }
    // set the individual row_pointers to point at the correct offsets of image_data
    for (int i = 0; i < height; ++i)
        row_pointers[height - 1 - i] = image_data + i * rowbytes;
    
    //read the png into image_data through row_pointers
    png_read_image(png_ptr, row_pointers);
    
    //Now generate the OpenGL texture object
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D,0, GL_RGB, width, height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*) image_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    
    //clean up memory and close stuff
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
    delete[] image_data;
    delete[] row_pointers;
    fclose(fp);
    
    return texture;
}