/*
Block match python wrapper
Meisam Rakhshanfar 2018
*/

#include <Python.h>
#include <thread>

static PyObject *GenError;

#define MAX_THREADS 64
#define BLOCK_SIZE 8
#define SEARCH_SIZE 8
#define LARGE_BSIZE (SEARCH_SIZE + BLOCK_SIZE)
#define OVERLAP_SIZE (SEARCH_SIZE / 2)
#define MAX_PIXEL_VALUE 255
#define PIXEL_BAD_VALUE 512
#define MAX_COST                                                               \
  (BLOCK_SIZE * BLOCK_SIZE * MAX_PIXEL_VALUE * MAX_PIXEL_VALUE * 4)
#define CH_NUM 5 // number of channels (matched blocks)
#define MIN_IMAGE_SIZE 64
#define MAX_IMAGE_SIZE 3840

// shift a block for half a block in x dimension
static void largeBlockShift(float *block) {
  for (int y = LARGE_BSIZE; y > 0; --y) {
    for (int x = LARGE_BSIZE - OVERLAP_SIZE; x > 0; --x) {
      *block = block[OVERLAP_SIZE];
      block++;
    }
    block += OVERLAP_SIZE;
  }
}

// find blocks with least MSE in a large (master) block
static void least_MSE_Blocks(const float *lblock, float *out, int cols, int xs,
                             int ys, int xe, int ye) {
  // out is (BLOCK_SIZE x BLOCK_SIZE x CH_NUM) pixels
  // without the reference block
  const int MATCHED_NUM = CH_NUM - 1;

  float ref_block[BLOCK_SIZE * BLOCK_SIZE];
  const float *lb = lblock + OVERLAP_SIZE * LARGE_BSIZE + OVERLAP_SIZE;
  float *sb = ref_block; // reference block
  // store the ref block into bmain
  for (int y = 0; y < BLOCK_SIZE; y++) {
    for (int x = 0; x < BLOCK_SIZE; x++) {
      *sb++ = *lb++;
    }
    lb += (2 * OVERLAP_SIZE);
  }

  float smin[MATCHED_NUM] = {MAX_COST, MAX_COST, MAX_COST, MAX_COST};
  int xmin[CH_NUM], ymin[CH_NUM];

  for (int y = 0; y <= SEARCH_SIZE; y++) {
    for (int x = 0; x <= SEARCH_SIZE; x++) {
      if (x == OVERLAP_SIZE && y == OVERLAP_SIZE)
        continue;

      sb = ref_block;
      lb = lblock + x * LARGE_BSIZE + y;
      float a;
      float s = 0;
      for (int xt = 0; xt < BLOCK_SIZE; xt++) {
        a = (*sb++) - (*lb++);
        s += a * a;
        a = (*sb++) - (*lb++);
        s += a * a;
        a = (*sb++) - (*lb++);
        s += a * a;
        a = (*sb++) - (*lb++);
        s += a * a;
        a = (*sb++) - (*lb++);
        s += a * a;
        a = (*sb++) - (*lb++);
        s += a * a;
        a = (*sb++) - (*lb++);
        s += a * a;
        a = (*sb++) - (*lb++);
        s += a * a;
        lb += SEARCH_SIZE;
      }
      // first 4 min MSE cost
      if (smin[0] > s) {
        xmin[3] = xmin[2];
        ymin[3] = ymin[2];
        smin[3] = smin[2];
        xmin[2] = xmin[1];
        ymin[2] = ymin[1];
        smin[2] = smin[1];
        xmin[1] = xmin[0];
        ymin[1] = ymin[0];
        smin[1] = smin[0];
        smin[0] = s;
        xmin[0] = x;
        ymin[0] = y;
      } else if (smin[1] > s) {
        xmin[3] = xmin[2];
        ymin[3] = ymin[2];
        smin[3] = smin[2];
        xmin[2] = xmin[1];
        ymin[2] = ymin[1];
        smin[2] = smin[1];
        smin[1] = s;
        xmin[1] = x;
        ymin[1] = y;
      } else if (smin[2] > s) {
        xmin[3] = xmin[2];
        ymin[3] = ymin[2];
        smin[3] = smin[2];
        smin[2] = s;
        xmin[2] = x;
        ymin[2] = y;
      } else if (smin[3] > s) {
        smin[3] = s;
        xmin[3] = x;
        ymin[3] = y;
      }
    }
  }

  // reference block
  xmin[4] = OVERLAP_SIZE;
  ymin[4] = OVERLAP_SIZE;

  for (int k = 0; k < CH_NUM; k++) {
    lb = lblock + xmin[k] * LARGE_BSIZE + ymin[k];

    for (int y = ys; y < ye; y++) {
      float *pxo = out + (y * cols + xs) * CH_NUM + k;
      const float *lbb = lb + y * LARGE_BSIZE + xs;

      for (int x = xs; x < xe; x++) {
        *pxo = *lbb++;
        pxo += CH_NUM;
      }
    }
  }
}

// fill the left part (0 to BLOCK_SIZE + OVERLAP_SIZE) of large block
static void fillBlockLeft(const float *img2D, float *large_block_out, int yc,
                          int rows, int cols) {
  int xc = 0 - OVERLAP_SIZE;
  int xe = xc + BLOCK_SIZE + OVERLAP_SIZE;
  int ys = yc - OVERLAP_SIZE;
  int ye = yc + BLOCK_SIZE + OVERLAP_SIZE;

  for (int y = ys; y < ye; ++y) {
    // out of image condition
    bool ccnd = false;
    if (y < 0 || y >= rows)
      ccnd = true;
    int ym = y * cols;
    for (int x = xc; x < xe; ++x) {
      if (x < 0 || x >= cols || ccnd)
        *large_block_out++ = PIXEL_BAD_VALUE;
      else
        *large_block_out++ = img2D[ym + x];
    }
    large_block_out += OVERLAP_SIZE;
  }
}

// fill the left part (BLOCK_SIZE + OVERLAP_SIZE to LARGE_BSIZE) of large block
static void fillBlockRight(const float *img2D, float *large_block_out, int xc,
                           int yc, int rows, int cols) {
  int xs = xc + BLOCK_SIZE;
  int xe = xc + BLOCK_SIZE + OVERLAP_SIZE;

  int ys = yc - OVERLAP_SIZE;
  int ye = yc + BLOCK_SIZE + OVERLAP_SIZE;

  for (int y = ys; y < ye; y++) {
    // out of image condition
    bool ccnd = false;
    if (y < 0 || y >= rows)
      ccnd = true;

    int ym = y * cols;

    // pointer to the right part of the large block
    large_block_out += (BLOCK_SIZE + OVERLAP_SIZE);

    for (int x = xs; x < xe; x++) {
      if (x < 0 || x >= cols || ccnd)
        *large_block_out++ = PIXEL_BAD_VALUE;
      else
        *large_block_out++ = img2D[ym + x];
    }
  }
}

// find matched block for a row of blocks
void matchedBlocksRow(float *dst, const float *src, int y, int img_row,
                      int img_col) {

  float lblock[LARGE_BSIZE * LARGE_BSIZE];

  int img_col_lim = img_col - BLOCK_SIZE;
  int img_row_lim = img_row - BLOCK_SIZE;

  fillBlockLeft(src, lblock, y, img_row, img_col);

  // center of (BLOCK_SIZE x BLOCK_SIZE) block starts from 2 to 6
  int ys = (OVERLAP_SIZE / 2);
  int ye = (OVERLAP_SIZE / 2) + OVERLAP_SIZE;
  if (y == 0)
    ys = 0;
  if (y == img_row_lim)
    ye = BLOCK_SIZE;

  for (int x = 0; x <= img_col_lim; x += OVERLAP_SIZE) {
    int xs = (OVERLAP_SIZE / 2);
    int xe = (OVERLAP_SIZE / 2) + OVERLAP_SIZE;

    if (x == 0)
      xs = 0;
    if (x == img_col_lim)
      xe = BLOCK_SIZE;

    fillBlockRight(src, lblock, x, y, img_row, img_col);
    least_MSE_Blocks(lblock, dst + (y * img_col + x) * CH_NUM, img_col, xs, ys,
                     xe, ye);
    largeBlockShift(lblock);
  }
}

int matchedBlocksThread(float *img_out, const float *img, int img_row,
                        int img_col, int thread_id, int thread_num) {

  int img_row_lim = img_row - BLOCK_SIZE;
  for (int y = 0; y <= img_row_lim; y += OVERLAP_SIZE) {
    if ((y / OVERLAP_SIZE) % thread_num == thread_id)
      matchedBlocksRow(img_out, img, y, img_row, img_col);
  }
  return 0;
}

int matchedBlocksImage(float *img_out, const float *img, int img_row,
                       int img_col, int thread_num) {

  std::thread t[MAX_THREADS];

  for (int i = 0; i < thread_num; ++i) {
    t[i] = std::thread(matchedBlocksThread, img_out, img, img_row, img_col, i,
                       thread_num);
  }
  for (int i = 0; i < thread_num; ++i) {
    t[i].join();
  }

  return 0;
}

PyObject *py_blockmatch(PyObject *self, PyObject *args) {
  PyObject *arg1, *arg2;
  Py_buffer b_imgin, b_imgout;
  int max_thread_num = -1;

  if (!PyArg_ParseTuple(args, "OO|i", &arg1, &arg2, &max_thread_num))
    return NULL;

  if (PyObject_GetBuffer(arg1, &b_imgin, PyBUF_FULL) < 0)
    return NULL;

  if (PyObject_GetBuffer(arg2, &b_imgout, PyBUF_FULL) < 0)
    return NULL;

  if (b_imgin.itemsize != 4 || b_imgout.itemsize != 4) {
    PyErr_SetString(GenError, "data type error (float 32 required)");
    return NULL;
  }
  if (b_imgin.ndim != 2 || b_imgout.ndim != 3) {
    PyErr_SetString(GenError,
                    "dimension type error (2d input and 3d output required)");
    return NULL;
  }

  int th_num = std::thread::hardware_concurrency();
  if (max_thread_num >= 1) {
    th_num = max_thread_num;
  }
  if (th_num > MAX_THREADS) {
    th_num = MAX_THREADS;
  }

  int img_row_i = (int)b_imgin.shape[0];
  int img_col_i = (int)b_imgin.shape[1];

  int img_row_o = (int)b_imgout.shape[0];
  int img_col_o = (int)b_imgout.shape[1];

  if (img_row_i != img_row_o || img_col_i != img_col_o ||
      (int)b_imgout.shape[2] != 5) {
    PyErr_SetString(GenError, "Output dimension error (5 required).\n");
    return NULL;
  }

  if ((b_imgin.strides[0] != b_imgin.shape[1] * b_imgout.itemsize) ||
      (b_imgout.strides[0] !=
       b_imgout.shape[1] * b_imgout.itemsize * b_imgout.shape[2]) ||
      (b_imgout.strides[1] != b_imgout.shape[2] * b_imgout.itemsize)) {
    PyErr_SetString(GenError, "Stride error.\n");
    return NULL;
  }

  if (img_row_i % 8 != 0 || img_col_i % 8 != 0) {
    PyErr_SetString(GenError, "dimensions should be divisible by 8.\n");
    return NULL;
  }

  if (img_row_i > MAX_IMAGE_SIZE || img_col_i > MAX_IMAGE_SIZE ||
      img_row_i < MIN_IMAGE_SIZE || img_col_i < MIN_IMAGE_SIZE) {
    PyErr_SetString(GenError, "Image size error too small or too large.\n");
    return NULL;
  }

  int results = matchedBlocksImage((float *)b_imgout.buf, (float *)b_imgin.buf,
                                   img_row_i, img_col_i, th_num);

  PyObject *res = PyLong_FromLong(results);
  PyBuffer_Release(&b_imgin);
  PyBuffer_Release(&b_imgout);

  return res;
}

PyMethodDef blockmatchMethods[] = {
    {"run", (PyCFunction)py_blockmatch, METH_VARARGS,
     "Block matching (image_in, image_out, max_threads)"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef blockmatchmodule = {
    PyModuleDef_HEAD_INIT, "blockmatch", /* name of module */
    "blockmatch Module C++",             /* module documentation, may be NULL */
    -1, /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
    blockmatchMethods};

PyMODINIT_FUNC PyInit_blockmatch(void) {
  PyObject *m = PyModule_Create(&blockmatchmodule);

  if (m == NULL)
    return NULL;

  GenError = PyErr_NewException("blockmatch.error", NULL, NULL);
  Py_INCREF(GenError);
  PyModule_AddObject(m, "error", GenError);

  return m;
}
