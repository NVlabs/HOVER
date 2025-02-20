import mujoco
import glfw
import numpy as np
import time
import pathlib
import yaml
from .callbacks import Callbacks

MUJOCO_VERSION = tuple(map(int, mujoco.__version__.split('.')))


class MujocoViewer(Callbacks):
    def __init__(
            self,
            model,
            data,
            mode='window',
            title="mujoco-python-viewer",
            width=None,
            height=None,
            hide_menus=False):
        super().__init__(hide_menus)

        self.model = model
        self.data = data
        self.render_mode = mode
        if self.render_mode not in ['offscreen', 'window']:
            raise NotImplementedError(
                "Invalid mode. Only 'offscreen' and 'window' are supported.")

        # keep true while running
        self.is_alive = True

        self.CONFIG_PATH = pathlib.Path.joinpath(
            pathlib.Path.home(), ".config/mujoco_viewer/config.yaml")

        # glfw init
        glfw.init()

        if not width:
            width, _ = glfw.get_video_mode(glfw.get_primary_monitor()).size

        if not height:
            _, height = glfw.get_video_mode(glfw.get_primary_monitor()).size

        if self.render_mode == 'offscreen':
            glfw.window_hint(glfw.VISIBLE, 0)

        self.window = glfw.create_window(
            width, height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
            self.window)

        # install callbacks only for 'window' mode
        if self.render_mode == 'window':
            window_width, _ = glfw.get_window_size(self.window)
            self._scale = framebuffer_width * 1.0 / window_width

            # set callbacks
            glfw.set_cursor_pos_callback(
                self.window, self._cursor_pos_callback)
            glfw.set_mouse_button_callback(
                self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            glfw.set_key_callback(self.window, self._key_callback)

        # create options, camera, scene, context
        self.vopt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.scn = mujoco.MjvScene(self.model, maxgeom=100000)
        self.pert = mujoco.MjvPerturb()

        self.ctx = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        width, height = glfw.get_framebuffer_size(self.window)

        # figures for creating 2D plots
        max_num_figs = 3
        self.figs = []
        width_adjustment = width % 4
        fig_w, fig_h = int(width / 4), int(height / 4)
        for idx in range(max_num_figs):
            fig = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(fig)
            fig.flg_extend = 1
            self.figs.append(fig)

        # load camera from configuration (if available)
        pathlib.Path(
            self.CONFIG_PATH.parent).mkdir(
            parents=True,
            exist_ok=True)
        pathlib.Path(self.CONFIG_PATH).touch(exist_ok=True)
        with open(self.CONFIG_PATH, "r") as f:
            try:
                cam_config = {
                    "type": self.cam.type,
                    "fixedcamid": self.cam.fixedcamid,
                    "trackbodyid": self.cam.trackbodyid,
                    "lookat": self.cam.lookat.tolist(),
                    "distance": self.cam.distance,
                    "azimuth": self.cam.azimuth,
                    "elevation": self.cam.elevation
                }
                load_config = yaml.safe_load(f)
                if isinstance(load_config, dict):
                    for key, val in load_config.items():
                        if key in cam_config.keys():
                            cam_config[key] = val
                if cam_config["type"] == mujoco.mjtCamera.mjCAMERA_FIXED:
                    if cam_config["fixedcamid"] < self.model.ncam:
                        self.cam.type = cam_config["type"]
                        self.cam.fixedcamid = cam_config["fixedcamid"]
                if cam_config["type"] == mujoco.mjtCamera.mjCAMERA_TRACKING:
                    if cam_config["trackbodyid"] < self.model.nbody:
                        self.cam.type = cam_config["type"]
                        self.cam.trackbodyid = cam_config["trackbodyid"]
                self.cam.lookat = np.array(cam_config["lookat"])
                self.cam.distance = cam_config["distance"]
                self.cam.azimuth = cam_config["azimuth"]
                self.cam.elevation = cam_config["elevation"]
            except yaml.YAMLError as e:
                print(e)

        # get viewport
        self.viewport = mujoco.MjrRect(
            0, 0, framebuffer_width, framebuffer_height)

        # overlay, markers
        self._overlay = {}
        self._markers_dict = {}

    def add_line_to_fig(self, line_name, fig_idx=0):
        assert isinstance(line_name, str), \
            "Line name must be a string."

        fig = self.figs[fig_idx]
        if line_name.encode('utf8') == b'':
            raise Exception(
                "Line name cannot be empty."
            )
        if line_name.encode('utf8') in fig.linename:
            raise Exception(
                "Line name already exists in this plot."
            )

        # this assumes all lines added by user have a non-empty name
        linecount = fig.linename.tolist().index(b'')

        # we want to add the line after the last non-empty index
        fig.linename[linecount] = line_name

        # assign x values
        for i in range(mujoco.mjMAXLINEPNT):
            fig.linedata[linecount][2*i] = -float(i)

    def add_data_to_line(self, line_name, line_data, fig_idx=0):
        fig = self.figs[fig_idx]

        try:
            _line_name = line_name.encode('utf8')
            linenames = fig.linename.tolist()
            line_idx = linenames.index(_line_name)
        except ValueError:
            raise Exception(
                "line name is not valid, add it to list before calling update"
            )

        pnt = min(mujoco.mjMAXLINEPNT, fig.linepnt[line_idx] + 1)
        # shift data
        for i in range(pnt-1, 0, -1):
            fig.linedata[line_idx][2*i + 1] = fig.linedata[line_idx][2*i - 1]

        # assign new
        fig.linepnt[line_idx] = pnt
        fig.linedata[line_idx][1] = line_data

    def add_marker(self, **marker_params):
        m_id = marker_params.get('id', 0)
        m_key = f'marker_{m_id}'
        self._markers_dict[m_key] = marker_params
    
    def draw_or_update_marker(self, ns: str, m_id: int,  **marker_params):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(
                'Ran out of geoms. maxgeom: %d' %
                self.scn.maxgeom)

        m_key = f'{ns}_{m_id}'
        if m_key not in self.marker_geoms:
            self.marker_geoms[m_key] = self.scn.geoms[self.scn.ngeom]
            self.scn.ngeom += 1
        
        m_geom = self.marker_geoms[m_key]
        m_geom.dataid = -1
        m_geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        m_geom.objid = -1
        m_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        m_geom.emission = 0
        m_geom.specular = 0.5
        m_geom.shininess = 0.5
        m_geom.reflectance = 0
        m_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        m_geom.size[:] = np.ones(3) * 0.1
        m_geom.mat[:] = np.eye(3)
        m_geom.rgba[:] = np.ones(4)

        for key, value in marker_params.items():
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(m_geom, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(m_geom, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    m_geom.label[0] = 0
                else:
                    m_geom.label = value
            elif hasattr(m_geom, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)))
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)
        return


    def _add_marker_to_scene(self, marker):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(
                'Ran out of geoms. maxgeom: %d' %
                self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]

        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR

        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if key == "id":
                continue

            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)))
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        self.scn.ngeom += 1

        return

    def _create_overlay(self):
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

        def add_overlay(gridpos, text1, text2):
            if gridpos not in self._overlay:
                self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1 + "\n"
            self._overlay[gridpos][1] += text2 + "\n"

        if self._render_every_frame:
            add_overlay(topleft, "", "")
        else:
            add_overlay(
                topleft,
                "Run speed = %.3f x real time" %
                self._run_speed,
                "[S]lower, [F]aster")
        add_overlay(
            topleft,
            "Ren[d]er every frame",
            "On" if self._render_every_frame else "Off")
        add_overlay(
            topleft, "Switch camera (#cams = %d)" %
            (self.model.ncam + 1), "[Tab] (camera ID = %d)" %
            self.cam.fixedcamid)
        add_overlay(
            topleft,
            "[C]ontact forces",
            "On" if self._contacts else "Off")
        add_overlay(
            topleft,
            "[J]oints",
            "On" if self._joints else "Off")
        add_overlay(
            topleft,
            "[G]raph Viewer",
            "Off" if self._hide_graph else "On")
        add_overlay(
            topleft,
            "[I]nertia",
            "On" if self._inertias else "Off")
        add_overlay(
            topleft,
            "Center of [M]ass",
            "On" if self._com else "Off")
        add_overlay(
            topleft, "Shad[O]ws", "On" if self._shadows else "Off"
        )
        add_overlay(
            topleft,
            "T[r]ansparent",
            "On" if self._transparent else "Off")
        add_overlay(
            topleft,
            "[W]ireframe",
            "On" if self._wire_frame else "Off")
        add_overlay(
            topleft,
            "Con[V]ex Hull Rendering",
            "On" if self._convex_hull_rendering else "Off",
        )
        if self._paused is not None:
            if not self._paused:
                add_overlay(topleft, "Stop", "[Space]")
            else:
                add_overlay(topleft, "Start", "[Space]")
                add_overlay(
                    topleft,
                    "Advance simulation by one step",
                    "[right arrow]")
        add_overlay(topleft, "Toggle geomgroup visibility (0-5)",
                    ",".join(["On" if g else "Off" for g in self.vopt.geomgroup]))
        add_overlay(
            topleft,
            "Referenc[e] frames",
            mujoco.mjtFrame(self.vopt.frame).name)
        add_overlay(topleft, "[H]ide Menus", "")
        if self._image_idx > 0:
            fname = self._image_path % (self._image_idx - 1)
            add_overlay(topleft, "Cap[t]ure frame", "Saved as %s" % fname)
        else:
            add_overlay(topleft, "Cap[t]ure frame", "")

        add_overlay(
            bottomleft, "FPS", "%d%s" %
            (1 / self._time_per_render, ""))

        if MUJOCO_VERSION >= (3, 0, 0):
            add_overlay(
                bottomleft, "Max solver iters", str(
                    max(self.data.solver_niter) + 1))
        else:
            add_overlay(
                bottomleft, "Solver iterations", str(
                    self.data.solver_iter + 1))

        add_overlay(
            bottomleft, "Step", str(
                round(
                    self.data.time / self.model.opt.timestep)))
        add_overlay(bottomleft, "timestep", "%.5f" % self.model.opt.timestep)

    def apply_perturbations(self):
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def read_pixels(self, camid=None, depth=False):
        if self.render_mode == 'window':
            raise NotImplementedError(
                "Use 'render()' in 'window' mode.")

        if camid is not None:
            if camid == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camid

        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
            self.window)
        # update scene
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scn)
        # render
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)
        shape = glfw.get_framebuffer_size(self.window)

        if depth:
            rgb_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            depth_img = np.zeros((shape[1], shape[0], 1), dtype=np.float32)
            mujoco.mjr_readPixels(rgb_img, depth_img, self.viewport, self.ctx)
            return (np.flipud(rgb_img), np.flipud(depth_img))
        else:
            img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, self.viewport, self.ctx)
            return np.flipud(img)

    def render(self):
        if self.render_mode == 'offscreen':
            raise NotImplementedError(
                "Use 'read_pixels()' for 'offscreen' mode.")
        if not self.is_alive:
            raise Exception(
                "GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()

            width, height = glfw.get_framebuffer_size(self.window)
            self.viewport.width, self.viewport.height = width, height

            with self._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn)
                # marker items
                for _ , marker in self._markers_dict.items():
                    self._add_marker_to_scene(marker)
                # render
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)
                # overlay items
                for gridpos, [t1, t2] in self._overlay.items():
                    menu_positions = [mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      mujoco.mjtGridPos.mjGRID_BOTTOMLEFT]
                    if gridpos in menu_positions and self._hide_menus:
                        continue

                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.ctx)

                # handle figures
                if not self._hide_graph:
                    for idx, fig in enumerate(self.figs):
                        width_adjustment = width % 4
                        x = int(3 * width / 4) + width_adjustment
                        y = idx * int(height / 4)
                        viewport = mujoco.MjrRect(
                            x, y, int(width / 4), int(height / 4))

                        has_lines = len([i for i in fig.linename if i != b''])
                        if has_lines:
                            mujoco.mjr_figure(viewport, fig, self.ctx)

                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + \
                0.1 * (time.time() - render_start)

            # clear overlay
            self._overlay.clear()

        if self._paused:
            while self._paused:
                update()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / \
                (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        # apply perturbation (should this come before mj_step?)
        self.apply_perturbations()

    def close(self):
        self.is_alive = False
        glfw.terminate()
        self.ctx.free()
