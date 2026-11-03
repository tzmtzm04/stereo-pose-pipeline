/*
 * Route2 display: NVMM NV12 fd → EGLImage → GL_TEXTURE_EXTERNAL_OES → X11 window.
 * B1: COCO-17 2D 骨架 GL overlay（无 D2H）。
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>

#include <nvbufsurface.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMaxKpts = 17;

// COCO-17 骨架（与 skeleton3d_view.COCO_SKELETON_17 一致）
constexpr int kSkeletonEdges[][2] = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},   {5, 6},  {5, 7},  {7, 9},
    {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15},
    {12, 14}, {14, 16},
};
constexpr int kNumSkeletonEdges =
    static_cast<int>(sizeof(kSkeletonEdges) / sizeof(kSkeletonEdges[0]));

PFNEGLCREATESYNCKHRPROC eglCreateSyncKHR_fn = nullptr;
PFNEGLDESTROYSYNCKHRPROC eglDestroySyncKHR_fn = nullptr;
PFNEGLCLIENTWAITSYNCKHRPROC eglClientWaitSyncKHR_fn = nullptr;
PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES_fn = nullptr;

bool init_egl_ext() {
    eglCreateSyncKHR_fn =
        (PFNEGLCREATESYNCKHRPROC)eglGetProcAddress("eglCreateSyncKHR");
    eglDestroySyncKHR_fn =
        (PFNEGLDESTROYSYNCKHRPROC)eglGetProcAddress("eglDestroySyncKHR");
    eglClientWaitSyncKHR_fn =
        (PFNEGLCLIENTWAITSYNCKHRPROC)eglGetProcAddress("eglClientWaitSyncKHR");
    glEGLImageTargetTexture2DOES_fn =
        (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress(
            "glEGLImageTargetTexture2DOES");
    return eglCreateSyncKHR_fn && eglDestroySyncKHR_fn &&
           eglClientWaitSyncKHR_fn && glEGLImageTargetTexture2DOES_fn;
}

struct Route2GlKpt {
    float x;
    float y;
    float conf;
    unsigned char has_depth;
};

struct Route2GlDisplay {
    Display* x_display = nullptr;
    Window x_window = 0;
    GC gc = nullptr;
    XFontStruct* font = nullptr;

    EGLDisplay egl_display = EGL_NO_DISPLAY;
    EGLContext egl_context = EGL_NO_CONTEXT;
    EGLSurface egl_surface = EGL_NO_SURFACE;
    EGLConfig egl_config = nullptr;

    GLuint texture_id = 0;
    GLuint program = 0;
    GLuint vbo = 0;
    GLint pos_location = -1;

    GLuint overlay_program = 0;
    GLint overlay_pos_loc = -1;
    GLint overlay_color_loc = -1;

    char status_text[512] = {};
    int win_w = 0;
    int win_h = 0;
    int base_w = 0;
    int base_h = 0;

    Route2GlKpt kpts[kMaxKpts]{};
    int num_kpts = 0;
    float conf_thresh = 0.25f;
    int video_w = 1920;
    int video_h = 1080;
    bool pose_enabled = false;
    int sbs_eye = 0;  // 0=full, 1=left, 2=right
    bool roi_zoom_enabled = false;
    float view_u0 = 0.f;
    float view_v0 = 0.f;
    float view_u1 = 1.f;
    float view_v1 = 1.f;
    bool dragging = false;
    int drag_x0 = 0;
    int drag_y0 = 0;
    int drag_x1 = 0;
    int drag_y1 = 0;
};

void eye_uv_range(const Route2GlDisplay* d, float* u0, float* v0, float* u1,
                  float* v1) {
    if (d->sbs_eye == 1) {
        *u0 = 0.f;
        *v0 = 0.f;
        *u1 = 0.5f;
        *v1 = 1.f;
    } else if (d->sbs_eye == 2) {
        *u0 = 0.5f;
        *v0 = 0.f;
        *u1 = 1.f;
        *v1 = 1.f;
    } else {
        *u0 = 0.f;
        *v0 = 0.f;
        *u1 = 1.f;
        *v1 = 1.f;
    }
}

void upload_video_verts(Route2GlDisplay* d) {
    float eu0 = 0.f;
    float ev0 = 0.f;
    float eu1 = 1.f;
    float ev1 = 1.f;
    eye_uv_range(d, &eu0, &ev0, &eu1, &ev1);
    const float u0 = eu0 + d->view_u0 * (eu1 - eu0);
    const float u1 = eu0 + d->view_u1 * (eu1 - eu0);
    const float v0 = ev0 + d->view_v0 * (ev1 - ev0);
    const float v1 = ev0 + d->view_v1 * (ev1 - ev0);
    const float verts[24] = {
        -1.f, -1.f, u0, v1, 1.f, -1.f, u1, v1, 1.f, 1.f, u1, v0,
        -1.f, -1.f, u0, v1, 1.f, 1.f,  u1, v0, -1.f, 1.f, u0, v0,
    };
    glBindBuffer(GL_ARRAY_BUFFER, d->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
}

void clamp_view_crop(Route2GlDisplay* d) {
    const float min_span = 0.02f;
    d->view_u0 = std::max(0.f, std::min(1.f, d->view_u0));
    d->view_v0 = std::max(0.f, std::min(1.f, d->view_v0));
    d->view_u1 = std::max(0.f, std::min(1.f, d->view_u1));
    d->view_v1 = std::max(0.f, std::min(1.f, d->view_v1));
    if (d->view_u1 < d->view_u0) {
        std::swap(d->view_u0, d->view_u1);
    }
    if (d->view_v1 < d->view_v0) {
        std::swap(d->view_v0, d->view_v1);
    }
    if (d->view_u1 - d->view_u0 < min_span) {
        const float c = 0.5f * (d->view_u0 + d->view_u1);
        d->view_u0 = std::max(0.f, c - min_span * 0.5f);
        d->view_u1 = std::min(1.f, d->view_u0 + min_span);
        d->view_u0 = d->view_u1 - min_span;
    }
    if (d->view_v1 - d->view_v0 < min_span) {
        const float c = 0.5f * (d->view_v0 + d->view_v1);
        d->view_v0 = std::max(0.f, c - min_span * 0.5f);
        d->view_v1 = std::min(1.f, d->view_v0 + min_span);
        d->view_v0 = d->view_v1 - min_span;
    }
}

void apply_view_crop(Route2GlDisplay* d, float u0, float v0, float u1, float v1) {
    d->view_u0 = u0;
    d->view_v0 = v0;
    d->view_u1 = u1;
    d->view_v1 = v1;
    clamp_view_crop(d);
    if (d->vbo) {
        upload_video_verts(d);
    }
}

void reset_view_crop(Route2GlDisplay* d) {
    apply_view_crop(d, 0.f, 0.f, 1.f, 1.f);
}

void zoom_view_crop(Route2GlDisplay* d, float factor) {
    const float cu = 0.5f * (d->view_u0 + d->view_u1);
    const float cv = 0.5f * (d->view_v0 + d->view_v1);
    const float hw = 0.5f * (d->view_u1 - d->view_u0) * factor;
    const float hh = 0.5f * (d->view_v1 - d->view_v0) * factor;
    apply_view_crop(d, cu - hw, cv - hh, cu + hw, cv + hh);
}

void apply_drag_selection(Route2GlDisplay* d, int x0, int y0, int x1, int y1) {
    if (d->win_w <= 0 || d->win_h <= 0) {
        return;
    }
    int sx0 = std::min(x0, x1);
    int sy0 = std::min(y0, y1);
    int sx1 = std::max(x0, x1);
    int sy1 = std::max(y0, y1);
    if (sx1 - sx0 < 8 || sy1 - sy0 < 8) {
        return;
    }
    const float rel_u0 = static_cast<float>(sx0) / static_cast<float>(d->win_w);
    const float rel_v0 = static_cast<float>(sy0) / static_cast<float>(d->win_h);
    const float rel_u1 = static_cast<float>(sx1) / static_cast<float>(d->win_w);
    const float rel_v1 = static_cast<float>(sy1) / static_cast<float>(d->win_h);
    const float span_u = d->view_u1 - d->view_u0;
    const float span_v = d->view_v1 - d->view_v0;
    apply_view_crop(
        d,
        d->view_u0 + rel_u0 * span_u,
        d->view_v0 + rel_v0 * span_v,
        d->view_u0 + rel_u1 * span_u,
        d->view_v0 + rel_v1 * span_v);
}

static const char kVertexShader[] =
    "attribute vec4 in_pos;\n"
    "varying vec2 interp_tc;\n"
    "void main() {\n"
    "  interp_tc = in_pos.zw;\n"
    "  gl_Position = vec4(in_pos.xy, 0.0, 1.0);\n"
    "}\n";

static const char kFragmentShader[] =
    "#extension GL_OES_EGL_image_external : require\n"
    "precision mediump float;\n"
    "varying vec2 interp_tc;\n"
    "uniform samplerExternalOES tex;\n"
    "void main() {\n"
    "  gl_FragColor = texture2D(tex, interp_tc);\n"
    "}\n";

static const char kOverlayVertexShader[] =
    "attribute vec2 a_pos;\n"
    "void main() {\n"
    "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "  gl_PointSize = 7.0;\n"
    "}\n";

static const char kOverlayFragmentShader[] =
    "precision mediump float;\n"
    "uniform vec3 u_color;\n"
    "void main() {\n"
    "  gl_FragColor = vec4(u_color, 1.0);\n"
    "}\n";

GLuint compile_shader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "route2_gl_display shader compile: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

int init_overlay_shaders(Route2GlDisplay* d) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, kOverlayVertexShader);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, kOverlayFragmentShader);
    if (!vs || !fs) {
        return -1;
    }
    d->overlay_program = glCreateProgram();
    glAttachShader(d->overlay_program, vs);
    glAttachShader(d->overlay_program, fs);
    glLinkProgram(d->overlay_program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint linked = GL_FALSE;
    glGetProgramiv(d->overlay_program, GL_LINK_STATUS, &linked);
    if (!linked) {
        return -1;
    }
    d->overlay_pos_loc =
        glGetAttribLocation(d->overlay_program, "a_pos");
    d->overlay_color_loc =
        glGetUniformLocation(d->overlay_program, "u_color");
    return 0;
}

int init_shaders(Route2GlDisplay* d) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, kVertexShader);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, kFragmentShader);
    if (!vs || !fs) {
        return -1;
    }
    d->program = glCreateProgram();
    glAttachShader(d->program, vs);
    glAttachShader(d->program, fs);
    glLinkProgram(d->program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint linked = GL_FALSE;
    glGetProgramiv(d->program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[4096];
        glGetProgramInfoLog(d->program, sizeof(log), nullptr, log);
        std::fprintf(stderr, "route2_gl_display link: %s\n", log);
        return -1;
    }
    glUseProgram(d->program);

    glGenBuffers(1, &d->vbo);
    upload_video_verts(d);

    d->pos_location = glGetAttribLocation(d->program, "in_pos");
    glBindBuffer(GL_ARRAY_BUFFER, d->vbo);
    glVertexAttribPointer(d->pos_location, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(d->pos_location);

    glActiveTexture(GL_TEXTURE0);
    GLint tex_loc = glGetUniformLocation(d->program, "tex");
    if (tex_loc >= 0) {
        glUniform1i(tex_loc, 0);
    }
    return init_overlay_shaders(d);
}

int init_egl(Route2GlDisplay* d) {
    static EGLint rgba8888[] = {EGL_RED_SIZE,   8, EGL_GREEN_SIZE, 8,
                                EGL_BLUE_SIZE,  8, EGL_ALPHA_SIZE, 8,
                                EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_NONE};

    d->egl_display = eglGetDisplay(d->x_display);
    if (d->egl_display == EGL_NO_DISPLAY) {
        return -1;
    }
    if (!eglInitialize(d->egl_display, nullptr, nullptr)) {
        return -1;
    }

    EGLint num_configs = 0;
    if (!eglChooseConfig(d->egl_display, rgba8888, &d->egl_config, 1,
                         &num_configs) ||
        num_configs < 1) {
        return -1;
    }

    EGLint ctx_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    d->egl_context = eglCreateContext(d->egl_display, d->egl_config,
                                      EGL_NO_CONTEXT, ctx_attribs);
    if (d->egl_context == EGL_NO_CONTEXT) {
        return -1;
    }

    d->egl_surface =
        eglCreateWindowSurface(d->egl_display, d->egl_config,
                               (EGLNativeWindowType)d->x_window, nullptr);
    if (d->egl_surface == EGL_NO_SURFACE) {
        return -1;
    }

    if (!eglMakeCurrent(d->egl_display, d->egl_surface, d->egl_surface,
                        d->egl_context)) {
        return -1;
    }
    eglSwapInterval(d->egl_display, 0);

    glEnable(GL_SCISSOR_TEST);
    glGenTextures(1, &d->texture_id);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, d->texture_id);

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    glScissor(viewport[0], viewport[1], viewport[2], viewport[3]);

    return init_shaders(d);
}

void update_gl_viewport(Route2GlDisplay* d, int width, int height) {
    if (!d || width <= 0 || height <= 0) {
        return;
    }
    d->win_w = width;
    d->win_h = height;
    if (d->egl_display != EGL_NO_DISPLAY &&
        d->egl_context != EGL_NO_CONTEXT) {
        eglMakeCurrent(d->egl_display, d->egl_surface, d->egl_surface,
                       d->egl_context);
        glViewport(0, 0, width, height);
        glScissor(0, 0, width, height);
    }
}

float px_to_ndc_x(float x, int video_w) {
    if (video_w <= 0) {
        return 0.f;
    }
    return 2.f * x / static_cast<float>(video_w) - 1.f;
}

float px_to_ndc_y(float y, int video_h) {
    if (video_h <= 0) {
        return 0.f;
    }
    return 1.f - 2.f * y / static_cast<float>(video_h);
}

// 与视频纹理同一套 view_crop（cover/ROI）：眼内像素 → 当前可见窗 NDC
bool kpt_to_ndc(const Route2GlDisplay* d, float px, float py, float* nx,
                float* ny) {
    if (!d || !nx || !ny || d->video_w <= 0 || d->video_h <= 0) {
        return false;
    }
    const float span_u = d->view_u1 - d->view_u0;
    const float span_v = d->view_v1 - d->view_v0;
    if (span_u <= 1e-6f || span_v <= 1e-6f) {
        return false;
    }
    const float u = px / static_cast<float>(d->video_w);
    const float v = py / static_cast<float>(d->video_h);
    const float u_rel = (u - d->view_u0) / span_u;
    const float v_rel = (v - d->view_v0) / span_v;
    // 裁切外的点不画（避免 cover 后骨架漂到窗外）
    if (u_rel < -0.02f || u_rel > 1.02f || v_rel < -0.02f || v_rel > 1.02f) {
        return false;
    }
    *nx = 2.f * u_rel - 1.f;
    *ny = 1.f - 2.f * v_rel;
    return true;
}

void draw_colored_lines(const std::vector<float>& verts, float r, float g,
                        float b, Route2GlDisplay* d) {
    if (verts.empty() || !d->overlay_program) {
        return;
    }
    glUseProgram(d->overlay_program);
    glUniform3f(d->overlay_color_loc, r, g, b);
    // 视频 VBO 仍绑定时，GLES 会把指针当 VBO 偏移，骨架顶点会读错。
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(d->overlay_pos_loc);
    glVertexAttribPointer(d->overlay_pos_loc, 2, GL_FLOAT, GL_FALSE, 0,
                          verts.data());
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(verts.size() / 2));
    glDisableVertexAttribArray(d->overlay_pos_loc);
}

void draw_colored_points(const std::vector<float>& verts, float r, float g,
                         float b, Route2GlDisplay* d) {
    if (verts.empty() || !d->overlay_program) {
        return;
    }
    glUseProgram(d->overlay_program);
    glUniform3f(d->overlay_color_loc, r, g, b);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(d->overlay_pos_loc);
    glVertexAttribPointer(d->overlay_pos_loc, 2, GL_FLOAT, GL_FALSE, 0,
                          verts.data());
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(verts.size() / 2));
    glDisableVertexAttribArray(d->overlay_pos_loc);
}

void draw_pose_overlay(Route2GlDisplay* d) {
    if (!d->pose_enabled || d->num_kpts <= 0) {
        return;
    }

    std::vector<float> line_verts;
    line_verts.reserve(kNumSkeletonEdges * 4);
    for (int e = 0; e < kNumSkeletonEdges; ++e) {
        const int i = kSkeletonEdges[e][0];
        const int j = kSkeletonEdges[e][1];
        if (i >= d->num_kpts || j >= d->num_kpts) {
            continue;
        }
        const Route2GlKpt& a = d->kpts[i];
        const Route2GlKpt& b = d->kpts[j];
        if (a.conf < d->conf_thresh || b.conf < d->conf_thresh) {
            continue;
        }
        float ax = 0.f;
        float ay = 0.f;
        float bx = 0.f;
        float by = 0.f;
        if (!kpt_to_ndc(d, a.x, a.y, &ax, &ay) ||
            !kpt_to_ndc(d, b.x, b.y, &bx, &by)) {
            continue;
        }
        line_verts.push_back(ax);
        line_verts.push_back(ay);
        line_verts.push_back(bx);
        line_verts.push_back(by);
    }

    std::vector<float> pts_plain;
    std::vector<float> pts_depth;
    pts_plain.reserve(d->num_kpts * 2);
    pts_depth.reserve(d->num_kpts * 2);
    for (int ki = 0; ki < d->num_kpts; ++ki) {
        const Route2GlKpt& k = d->kpts[ki];
        if (k.conf < d->conf_thresh) {
            continue;
        }
        float nx = 0.f;
        float ny = 0.f;
        if (!kpt_to_ndc(d, k.x, k.y, &nx, &ny)) {
            continue;
        }
        if (k.has_depth) {
            pts_depth.push_back(nx);
            pts_depth.push_back(ny);
        } else {
            pts_plain.push_back(nx);
            pts_plain.push_back(ny);
        }
    }

    glDisable(GL_SCISSOR_TEST);
    if (d->pos_location >= 0) {
        glDisableVertexAttribArray(d->pos_location);
    }
    // 骨架线 BGR(0,255,128) → RGB
    draw_colored_lines(line_verts, 0.f, 1.f, 0.5f, d);

    // 普通点 BGR(0,255,0) → RGB
    draw_colored_points(pts_plain, 0.f, 1.f, 0.f, d);
    // 有深度点 BGR(0,200,255) → RGB
    draw_colored_points(pts_depth, 1.f, 0.784f, 0.f, d);
    glEnable(GL_SCISSOR_TEST);
}

void draw_status(Route2GlDisplay* d) {
    if (!d->status_text[0] || !d->gc || !d->font) {
        return;
    }
    const int bar_h = 28;
    const int y = d->win_h - 8;
    XSetForeground(d->x_display, d->gc,
                   BlackPixel(d->x_display, DefaultScreen(d->x_display)));
    XFillRectangle(d->x_display, d->x_window, d->gc, 0, d->win_h - bar_h,
                   d->win_w, bar_h);
    XSetForeground(d->x_display, d->gc,
                   WhitePixel(d->x_display, DefaultScreen(d->x_display)));
    XSetFont(d->x_display, d->gc, d->font->fid);
    XDrawString(d->x_display, d->x_window, d->gc, 8, y, d->status_text,
                (int)std::strlen(d->status_text));
    XFlush(d->x_display);
}

}  // namespace

extern "C" {

void route2_gl_display_destroy(Route2GlDisplay* d);

int route2_gl_display_available(void) {
    return init_egl_ext() ? 1 : 0;
}

static void ensure_x_threads() {
    static bool inited = false;
    if (!inited) {
        // Host(kiosk) 与 UI 线程都会碰 X11；必须先开多线程
        XInitThreads();
        inited = true;
    }
}

Route2GlDisplay* route2_gl_display_create_ex(const char* title, int width,
                                             int height, int start_mapped,
                                             int override_redirect, int origin_x,
                                             int origin_y) {
    if (!init_egl_ext()) {
        std::fprintf(stderr, "route2_gl_display: EGL/GL extensions missing\n");
        return nullptr;
    }
    ensure_x_threads();

    auto* d = new Route2GlDisplay();
    d->win_w = width > 0 ? width : 1920;
    d->win_h = height > 0 ? height : 1080;
    d->base_w = d->win_w;
    d->base_h = d->win_h;

    d->x_display = XOpenDisplay(nullptr);
    if (!d->x_display) {
        std::fprintf(stderr, "route2_gl_display: XOpenDisplay failed\n");
        delete d;
        return nullptr;
    }

    const int screen = DefaultScreen(d->x_display);
    XSetWindowAttributes attrs{};
    attrs.background_pixel = BlackPixel(d->x_display, screen);
    attrs.override_redirect = override_redirect ? 1 : 0;
    attrs.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask |
                       ButtonPressMask | ButtonReleaseMask | ButtonMotionMask;

    // origin 可放屏外（kiosk 预热），避免 eglSwap 把未定位窗弹到 (0,0)
    d->x_window = XCreateWindow(
        d->x_display, DefaultRootWindow(d->x_display), origin_x, origin_y,
        d->win_w, d->win_h, 0, DefaultDepth(d->x_display, screen), InputOutput,
        CopyFromParent, CWBackPixel | CWOverrideRedirect | CWEventMask, &attrs);

    if (title && title[0]) {
        XStoreName(d->x_display, d->x_window, title);
    }
    if (start_mapped) {
        XMapWindow(d->x_display, d->x_window);
    }
    XFlush(d->x_display);

    d->gc = XCreateGC(d->x_display, d->x_window, 0, nullptr);
    d->font = XLoadQueryFont(d->x_display, "9x15bold");

    if (init_egl(d) != 0) {
        std::fprintf(stderr, "route2_gl_display: EGL init failed\n");
        route2_gl_display_destroy(d);
        return nullptr;
    }
    return d;
}

Route2GlDisplay* route2_gl_display_create(const char* title, int width,
                                          int height) {
    return route2_gl_display_create_ex(title, width, height, /*start_mapped=*/1,
                                       /*override_redirect=*/0, /*origin_x=*/0,
                                       /*origin_y=*/0);
}

void route2_gl_display_destroy(Route2GlDisplay* d) {
    if (!d) {
        return;
    }
    if (d->texture_id) {
        glDeleteTextures(1, &d->texture_id);
    }
    if (d->vbo) {
        glDeleteBuffers(1, &d->vbo);
    }
    if (d->program) {
        glDeleteProgram(d->program);
    }
    if (d->overlay_program) {
        glDeleteProgram(d->overlay_program);
    }
    if (d->egl_display != EGL_NO_DISPLAY) {
        eglMakeCurrent(d->egl_display, EGL_NO_SURFACE, EGL_NO_CONTEXT,
                       EGL_NO_CONTEXT);
    }
    if (d->egl_surface != EGL_NO_SURFACE) {
        eglDestroySurface(d->egl_display, d->egl_surface);
    }
    if (d->egl_context != EGL_NO_CONTEXT) {
        eglDestroyContext(d->egl_display, d->egl_context);
    }
    if (d->egl_display != EGL_NO_DISPLAY) {
        eglTerminate(d->egl_display);
    }
    if (d->font) {
        XFreeFont(d->x_display, d->font);
    }
    if (d->gc) {
        XFreeGC(d->x_display, d->gc);
    }
    if (d->x_window) {
        XDestroyWindow(d->x_display, d->x_window);
    }
    if (d->x_display) {
        XCloseDisplay(d->x_display);
    }
    delete d;
}

void route2_gl_display_set_status(Route2GlDisplay* d, const char* text) {
    if (!d) {
        return;
    }
    if (!text) {
        d->status_text[0] = '\0';
        return;
    }
    std::snprintf(d->status_text, sizeof(d->status_text), "%s", text);
}

void route2_gl_display_clear_pose(Route2GlDisplay* d) {
    if (!d) {
        return;
    }
    d->pose_enabled = false;
    d->num_kpts = 0;
}

void route2_gl_display_set_sbs_eye(Route2GlDisplay* d, int eye) {
    if (!d) {
        return;
    }
    if (eye < 0) {
        eye = 0;
    } else if (eye > 2) {
        eye = 2;
    }
    d->sbs_eye = eye;
    if (d->vbo) {
        upload_video_verts(d);
    }
}

void route2_gl_display_set_sbs_left_eye(Route2GlDisplay* d, int enabled) {
    route2_gl_display_set_sbs_eye(d, enabled ? 1 : 0);
}

void route2_gl_display_set_pose(
    Route2GlDisplay* d,
    const Route2GlKpt* kpts,
    int num_kpts,
    float conf_thresh,
    int video_w,
    int video_h) {
    if (!d) {
        return;
    }
    d->pose_enabled = (kpts != nullptr && num_kpts > 0);
    d->num_kpts = 0;
    if (!d->pose_enabled) {
        return;
    }
    const int n = num_kpts > kMaxKpts ? kMaxKpts : num_kpts;
    for (int i = 0; i < n; ++i) {
        d->kpts[i] = kpts[i];
    }
    d->num_kpts = n;
    d->conf_thresh = conf_thresh;
    d->video_w = video_w > 0 ? video_w : 1920;
    d->video_h = video_h > 0 ? video_h : 1080;
}

void route2_gl_display_enable_roi_zoom(Route2GlDisplay* d, int enabled) {
    if (!d) {
        return;
    }
    d->roi_zoom_enabled = enabled != 0;
}

void route2_gl_display_reset_view(Route2GlDisplay* d) {
    if (!d) {
        return;
    }
    reset_view_crop(d);
}

void route2_gl_display_set_view_crop(Route2GlDisplay* d, float u0, float v0,
                                     float u1, float v1) {
    if (!d) {
        return;
    }
    apply_view_crop(d, u0, v0, u1, v1);
}

void route2_gl_display_get_view_crop(Route2GlDisplay* d, float* u0, float* v0,
                                     float* u1, float* v1) {
    if (!d) {
        return;
    }
    if (u0) {
        *u0 = d->view_u0;
    }
    if (v0) {
        *v0 = d->view_v0;
    }
    if (u1) {
        *u1 = d->view_u1;
    }
    if (v1) {
        *v1 = d->view_v1;
    }
}

void route2_gl_display_set_window_size(Route2GlDisplay* d, int width,
                                       int height) {
    if (!d || !d->x_display || !d->x_window || width <= 0 || height <= 0) {
        return;
    }
    update_gl_viewport(d, width, height);
    XResizeWindow(d->x_display, d->x_window, width, height);
    XFlush(d->x_display);
}

void route2_gl_display_set_window_position(Route2GlDisplay* d, int x, int y) {
    if (!d || !d->x_display || !d->x_window) {
        return;
    }
    XMoveWindow(d->x_display, d->x_window, x, y);
    XFlush(d->x_display);
}

void route2_gl_display_set_window_geometry(Route2GlDisplay* d, int x, int y,
                                           int width, int height) {
    if (!d || !d->x_display || !d->x_window || width <= 0 || height <= 0) {
        return;
    }
    update_gl_viewport(d, width, height);
    XMoveResizeWindow(d->x_display, d->x_window, x, y,
                      static_cast<unsigned int>(width),
                      static_cast<unsigned int>(height));
    XFlush(d->x_display);
}

void route2_gl_display_get_window_geometry(Route2GlDisplay* d, int* x, int* y,
                                           int* width, int* height) {
    if (!d || !d->x_display || !d->x_window) {
        if (x) {
            *x = 0;
        }
        if (y) {
            *y = 0;
        }
        if (width) {
            *width = 0;
        }
        if (height) {
            *height = 0;
        }
        return;
    }
    Window root = 0;
    int wx = 0;
    int wy = 0;
    unsigned int ww = 0;
    unsigned int wh = 0;
    unsigned int border = 0;
    unsigned int depth = 0;
    if (!XGetGeometry(d->x_display, d->x_window, &root, &wx, &wy, &ww, &wh,
                      &border, &depth)) {
        if (x) {
            *x = 0;
        }
        if (y) {
            *y = 0;
        }
        if (width) {
            *width = d->win_w;
        }
        if (height) {
            *height = d->win_h;
        }
        return;
    }
    if (x) {
        *x = wx;
    }
    if (y) {
        *y = wy;
    }
    if (width) {
        *width = static_cast<int>(ww);
    }
    if (height) {
        *height = static_cast<int>(wh);
    }
}

void route2_gl_display_set_window_mapped(Route2GlDisplay* d, int mapped) {
    if (!d || !d->x_display || !d->x_window) {
        return;
    }
    if (mapped) {
        XMapWindow(d->x_display, d->x_window);
    } else {
        XUnmapWindow(d->x_display, d->x_window);
    }
    XFlush(d->x_display);
}

void route2_gl_display_set_override_redirect(Route2GlDisplay* d, int enabled) {
    // 绕过 WM 层叠：盖过 Firefox --kiosk 全屏（需在 unmap 状态下改）
    if (!d || !d->x_display || !d->x_window) {
        return;
    }
    XUnmapWindow(d->x_display, d->x_window);
    XSetWindowAttributes attrs{};
    attrs.override_redirect = enabled ? 1 : 0;
    XChangeWindowAttributes(d->x_display, d->x_window, CWOverrideRedirect,
                            &attrs);
    XFlush(d->x_display);
}

void route2_gl_display_set_window_above(Route2GlDisplay* d, int above) {
    if (!d || !d->x_display || !d->x_window) {
        return;
    }
    Atom wm_state = XInternAtom(d->x_display, "_NET_WM_STATE", False);
    Atom above_atom =
        XInternAtom(d->x_display, "_NET_WM_STATE_ABOVE", False);
    XEvent ev{};
    ev.xclient.type = ClientMessage;
    ev.xclient.window = d->x_window;
    ev.xclient.message_type = wm_state;
    ev.xclient.format = 32;
    ev.xclient.data.l[0] = above ? 1 : 0;  // _NET_WM_STATE_ADD / REMOVE
    ev.xclient.data.l[1] = static_cast<long>(above_atom);
    ev.xclient.data.l[2] = 0;
    ev.xclient.data.l[3] = 1;
    XSendEvent(d->x_display, DefaultRootWindow(d->x_display), False,
               SubstructureRedirectMask | SubstructureNotifyMask, &ev);
    if (above) {
        XRaiseWindow(d->x_display, d->x_window);
    }
    XFlush(d->x_display);
}

void route2_gl_display_raise(Route2GlDisplay* d) {
    if (!d || !d->x_display || !d->x_window) {
        return;
    }
    XRaiseWindow(d->x_display, d->x_window);
    XFlush(d->x_display);
}

void route2_gl_display_get_window_size(Route2GlDisplay* d, int* width,
                                       int* height) {
    if (!d) {
        return;
    }
    if (width) {
        *width = d->win_w;
    }
    if (height) {
        *height = d->win_h;
    }
}

double route2_gl_display_render_fd(Route2GlDisplay* d, int fd) {
    if (!d || fd < 0) {
        return -1.0;
    }
    const auto t0 = Clock::now();

    NvBufSurface* nvbuf = nullptr;
    if (NvBufSurfaceFromFd(fd, (void**)&nvbuf) != 0 || !nvbuf) {
        std::fprintf(stderr, "route2_gl_display: NvBufSurfaceFromFd failed\n");
        return -1.0;
    }
    if (NvBufSurfaceMapEglImage(nvbuf, 0) != 0) {
        std::fprintf(stderr, "route2_gl_display: MapEglImage failed\n");
        return -1.0;
    }

    EGLImageKHR egl_image = nvbuf->surfaceList[0].mappedAddr.eglImage;
    if (!egl_image) {
        NvBufSurfaceUnMapEglImage(nvbuf, 0);
        std::fprintf(stderr, "route2_gl_display: null EGLImage\n");
        return -1.0;
    }

    glUseProgram(d->program);
    glEnableVertexAttribArray(d->pos_location);
    glBindBuffer(GL_ARRAY_BUFFER, d->vbo);
    glVertexAttribPointer(d->pos_location, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, d->texture_id);
    glEGLImageTargetTexture2DOES_fn(GL_TEXTURE_EXTERNAL_OES, egl_image);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    if (glGetError() != GL_NO_ERROR) {
        NvBufSurfaceUnMapEglImage(nvbuf, 0);
        return -1.0;
    }

    draw_pose_overlay(d);

    EGLSyncKHR sync =
        eglCreateSyncKHR_fn(d->egl_display, EGL_SYNC_FENCE_KHR, nullptr);
    if (sync != EGL_NO_SYNC_KHR) {
        eglClientWaitSyncKHR_fn(d->egl_display, sync,
                                EGL_SYNC_FLUSH_COMMANDS_BIT_KHR,
                                3000000);
        eglDestroySyncKHR_fn(d->egl_display, sync);
    }

    eglSwapBuffers(d->egl_display, d->egl_surface);
    NvBufSurfaceUnMapEglImage(nvbuf, 0);
    draw_status(d);

    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return ms;
}

int route2_gl_display_poll_quit(Route2GlDisplay* d) {
    if (!d || !d->x_display) {
        return 0;
    }
    while (XPending(d->x_display) > 0) {
        XEvent ev{};
        XNextEvent(d->x_display, &ev);
        if (ev.type == ConfigureNotify &&
            ev.xconfigure.window == d->x_window) {
            update_gl_viewport(d, ev.xconfigure.width, ev.xconfigure.height);
        } else if (ev.type == KeyPress) {
            KeySym sym = XLookupKeysym(&ev.xkey, 0);
            if (sym == XK_q || sym == XK_Q) {
                return 1;
            }
            if (!d->roi_zoom_enabled) {
                continue;
            }
            if (sym == XK_equal || sym == XK_plus || sym == XK_KP_Add) {
                zoom_view_crop(d, 0.85f);
            } else if (sym == XK_minus || sym == XK_KP_Subtract ||
                       sym == XK_underscore) {
                zoom_view_crop(d, 1.18f);
            } else if (sym == XK_0 || sym == XK_KP_0 || sym == XK_r ||
                       sym == XK_R) {
                reset_view_crop(d);
            }
        } else if (d->roi_zoom_enabled && ev.type == ButtonPress &&
                   ev.xbutton.button == Button1) {
            d->dragging = true;
            d->drag_x0 = ev.xbutton.x;
            d->drag_y0 = ev.xbutton.y;
            d->drag_x1 = ev.xbutton.x;
            d->drag_y1 = ev.xbutton.y;
        } else if (d->roi_zoom_enabled && ev.type == ButtonRelease &&
                   ev.xbutton.button == Button1 && d->dragging) {
            d->dragging = false;
            apply_drag_selection(d, d->drag_x0, d->drag_y0, d->drag_x1,
                                 d->drag_y1);
        } else if (d->roi_zoom_enabled && ev.type == MotionNotify &&
                   d->dragging) {
            d->drag_x1 = ev.xmotion.x;
            d->drag_y1 = ev.xmotion.y;
        }
    }
    return 0;
}

}  // extern "C"
