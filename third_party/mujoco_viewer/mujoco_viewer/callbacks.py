import mujoco
import glfw
import numpy as np
import imageio
import yaml
from threading import Lock

MUJOCO_VERSION = tuple(map(int, mujoco.__version__.split('.')))


class Callbacks:
    def __init__(self, hide_menus):
        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        self._last_left_click_time = None
        self._last_right_click_time = None
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._paused = False
        self._hide_graph = False
        self._transparent = False
        self._contacts = False
        self._joints = False
        self._shadows = True
        self._wire_frame = False
        self._convex_hull_rendering = False
        self._inertias = False
        self._com = False
        self._render_every_frame = True
        self._image_idx = 0
        self._image_path = "/tmp/frame_%07d.png"
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False
        self._hide_menus = hide_menus

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            if key == glfw.KEY_LEFT_ALT:
                self._hide_menus = False
            return
        # Switch cameras
        elif key == glfw.KEY_TAB:
            self.cam.fixedcamid += 1
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            if self.cam.fixedcamid >= self.model.ncam:
                self.cam.fixedcamid = -1
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Pause simulation
        elif key == glfw.KEY_SPACE and self._paused is not None:
            self._paused = not self._paused
        # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        # Slows down simulation
        elif key == glfw.KEY_S and mods != glfw.MOD_CONTROL:
            self._run_speed /= 2.0
        # Speeds up simulation
        elif key == glfw.KEY_F:
            self._run_speed *= 2.0
        # Turn off / turn on rendering every frame.
        elif key == glfw.KEY_D:
            self._render_every_frame = not self._render_every_frame
        # Capture screenshot
        elif key == glfw.KEY_T:
            img = np.zeros(
                (glfw.get_framebuffer_size(
                    self.window)[1], glfw.get_framebuffer_size(
                    self.window)[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, self.viewport, self.ctx)
            imageio.imwrite(self._image_path % self._image_idx, np.flipud(img))
            self._image_idx += 1
        # Display contact forces
        elif key == glfw.KEY_C:
            self._contacts = not self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self._contacts
        elif key == glfw.KEY_J:
            self._joints = not self._joints
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = self._joints
        # Display mjtFrame
        elif key == glfw.KEY_E:
            self.vopt.frame += 1
            if self.vopt.frame == mujoco.mjtFrame.mjNFRAME.value:
                self.vopt.frame = 0
        # Hide overlay menu
        elif key == glfw.KEY_LEFT_ALT:
            self._hide_menus = True
        elif key == glfw.KEY_H:
            self._hide_menus = not self._hide_menus
        # Make transparent
        elif key == glfw.KEY_R:
            self._transparent = not self._transparent
            if self._transparent:
                self.model.geom_rgba[:, 3] /= 5.0
            else:
                self.model.geom_rgba[:, 3] *= 5.0
        # Toggle Graph overlay
        elif key == glfw.KEY_G:
            self._hide_graph = not self._hide_graph
        # Display inertia
        elif key == glfw.KEY_I:
            self._inertias = not self._inertias
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = self._inertias
        # Display center of mass
        elif key == glfw.KEY_M:
            self._com = not self._com
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_COM] = self._com
        # Shadow Rendering
        elif key == glfw.KEY_O:
            self._shadows = not self._shadows
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self._shadows
        # Convex-Hull rendering
        elif key == glfw.KEY_V:
            self._convex_hull_rendering = not self._convex_hull_rendering
            self.vopt.flags[
                mujoco.mjtVisFlag.mjVIS_CONVEXHULL
            ] = self._convex_hull_rendering
        # Wireframe Rendering
        elif key == glfw.KEY_W:
            self._wire_frame = not self._wire_frame
            self.scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = self._wire_frame
        # Geom group visibility
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5):
            self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        elif key == glfw.KEY_S and mods == glfw.MOD_CONTROL:
            cam_config = {
                "type": self.cam.type,
                "fixedcamid": self.cam.fixedcamid,
                "trackbodyid": self.cam.trackbodyid,
                "lookat": self.cam.lookat.tolist(),
                "distance": self.cam.distance,
                "azimuth": self.cam.azimuth,
                "elevation": self.cam.elevation
            }
            try:
                with open(self.CONFIG_PATH, "w") as f:
                    yaml.dump(cam_config, f)
                print("Camera config saved at {}".format(self.CONFIG_PATH))
            except Exception as e:
                print(e)
        # Quit
        if key == glfw.KEY_ESCAPE:
            print("Pressed ESC")
            print("Quitting.")
            glfw.set_window_should_close(self.window, True)
        return

    def _cursor_pos_callback(self, window, xpos, ypos):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self._button_right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            if self.pert.active:
                mujoco.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.pert)
            else:
                mujoco.mjv_moveCamera(
                    self.model,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.cam)

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        # detect a left- or right- doubleclick
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        time_now = glfw.get_time()

        if self._button_left_pressed:
            if self._last_left_click_time is None:
                self._last_left_click_time = glfw.get_time()

            time_diff = (time_now - self._last_left_click_time)
            if time_diff > 0.01 and time_diff < 0.3:
                self._left_double_click_pressed = True
            self._last_left_click_time = time_now

        if self._button_right_pressed:
            if self._last_right_click_time is None:
                self._last_right_click_time = glfw.get_time()

            time_diff = (time_now - self._last_right_click_time)
            if time_diff > 0.01 and time_diff < 0.2:
                self._right_double_click_pressed = True
            self._last_right_click_time = time_now

        # set perturbation
        key = mods == glfw.MOD_CONTROL
        newperturb = 0
        if key and self.pert.select > 0:
            # right: translate, left: rotate
            if self._button_right_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_ROTATE

            # perturbation onste: reset reference
            if newperturb and not self.pert.active:
                mujoco.mjv_initPerturb(
                    self.model, self.data, self.scn, self.pert)
        self.pert.active = newperturb

        # handle doubleclick
        if self._left_double_click_pressed or self._right_double_click_pressed:
            # determine selection mode
            selmode = 0
            if self._left_double_click_pressed:
                selmode = 1
            if self._right_double_click_pressed:
                selmode = 2
            if self._right_double_click_pressed and key:
                selmode = 3

            # find geom and 3D click point, get corresponding body
            width, height = self.viewport.width, self.viewport.height
            aspectratio = width / height
            relx = x / width
            rely = (self.viewport.height - y) / height
            selpnt = np.zeros((3, 1), dtype=np.float64)
            selgeom = np.zeros((1, 1), dtype=np.int32)
            selflex = np.zeros((1, 1), dtype=np.int32)
            selskin = np.zeros((1, 1), dtype=np.int32)

            if MUJOCO_VERSION >= (3, 0, 0):
                selbody = mujoco.mjv_select(
                    self.model,
                    self.data,
                    self.vopt,
                    aspectratio,
                    relx,
                    rely,
                    self.scn,
                    selpnt,
                    selgeom,
                    selflex,
                    selskin)
            else:
                selbody = mujoco.mjv_select(
                    self.model,
                    self.data,
                    self.vopt,
                    aspectratio,
                    relx,
                    rely,
                    self.scn,
                    selpnt,
                    selgeom,
                    selskin)

            # set lookat point, start tracking is requested
            if selmode == 2 or selmode == 3:
                # set cam lookat
                if selbody >= 0:
                    self.cam.lookat = selpnt.flatten()
                # switch to tracking camera if dynamic body clicked
                if selmode == 3 and selbody > 0:
                    self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    self.cam.trackbodyid = selbody
                    self.cam.fixedcamid = -1
            # set body selection
            else:
                if selbody >= 0:
                    # record selection
                    self.pert.select = selbody
                    self.pert.skinselect = selskin
                    # compute localpos
                    vec = selpnt.flatten() - self.data.xpos[selbody]
                    mat = self.data.xmat[selbody].reshape(3, 3)
                    self.pert.localpos = self.data.xmat[selbody].reshape(
                        3, 3).dot(vec)
                else:
                    self.pert.select = 0
                    self.pert.skinselect = -1
            # stop perturbation on select
            self.pert.active = 0

        # 3D release
        if act == glfw.RELEASE:
            self.pert.active = 0

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.scn, self.cam)