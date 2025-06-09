import time
import numpy as np
import cv2
import pyrealsense2 as rs
import mediapipe as mp
import redis



SCRUB_POINTS_KEY = "sai::commands::Sponge::scrub_points"

# --- MediaPipe setup ---
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose()
TORSO_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP
]


def plane_orientation(p1, p2, p3):
    """
    Given three 3D points on a plane:
      p1 = top_left, p2 = top_right, p3 = bottom_mid
    Returns (roll, pitch, yaw) in radians using ZYX convention.
    """
    print(" p1:", p1, "   p2:", p2, "   p3:", p3)
    v1 = np.array(p2) - np.array(p1)   # along top edge
    v2 = np.array(p3) - np.array(p1)
    ex = v1 / np.linalg.norm(v1)       # x‐axis
    ez = np.cross(v2, v1)
    ez = ez / np.linalg.norm(ez)       # z‐axis (normal)
    ey = np.cross(ez, ex)              # y‐axis
    R = np.column_stack((ex, ey, ez))  # rotation matrix
    # ZYX Euler (yaw, pitch, roll):
    roll  = np.arctan2(R[2,1], R[2,2])
    pitch = np.arcsin(-R[2,0])
    yaw   = np.arctan2(R[1,0], R[0,0])
    return roll, pitch, yaw


def interpolate_grid(top_left, top_right, bottom_left, bottom_right, n, m):
    grid = []
    for i in range(m):
        v = i/(m-1) if m>1 else 0
        row = []
        for j in range(n):
            u = j/(n-1) if n>1 else 0
            top    = np.array(top_left)*(1-u) + np.array(top_right)*u
            bottom = np.array(bottom_left)*(1-u) + np.array(bottom_right)*u
            pt     = (1-v)*top + v*bottom
            row.append((pt[0], pt[1]))
        grid.append(row)
    return grid


def publish_all_as_one(true_points, orientations,
                       host="localhost", port=6379, db=0):
    # build the two semicolon-delimited lists
    pos_str = ";".join(f"{X:.6f},{Y:.6f},{Z:.6f}"
                       for X, Y, Z in true_points)
    ori_str = ";".join(f"{r:.6f},{p:.6f},{y:.6f}"
                       for r, p, y in orientations)    # single blob:  "<x,y,z>;…;<x,y,z>|<r,p,y>;…;<r,p,y>"
    blob = pos_str + "|" + ori_str    
    r = redis.Redis(host=host, port=port, db=db, socket_timeout=2)
    r.set(SCRUB_POINTS_KEY, blob)
    print(f"[publish] SET {SCRUB_POINTS_KEY} size={len(blob)}")
    


def point_transform(depth_frame, depth_intrin, w, h, grid, ri, ci):
    px, py = grid[ri][ci]
    xi, yi = int(np.clip(px, 0, w-1)), int(np.clip(py, 0, h-1))
    depth = depth_frame.get_distance(xi, yi)
    camera_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [xi, yi], depth)
    θ1 = np.pi / 2
    Rx1 = np.array([
        [1, 0, 0],
        [0, np.cos(θ1), -np.sin(θ1)],
        [0, np.sin(θ1),  np.cos(θ1)]
    ])
    θ2 = np.pi
    Rz = np.array([
        [np.cos(θ2), -np.sin(θ2), 0],
        [np.sin(θ2),  np.cos(θ2), 0],
        [0, 0, 1]
    ])
    R = Rx1 @ Rz
    #x_tweak = 0.015
    x_tweak = -0.00 - 0.30 # was 0.31
    z_tweak = 0.01 
    transformed_point = (R @ np.array(camera_point)) + np.array((0.835014 + x_tweak, 0.878464, 0.08324 + z_tweak ))
    return xi, yi, transformed_point


def get_torso_grid_with_depth(num_columns=9, num_rows=13, wait_time=3):

    # — RealSense setup & alignment —
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    aligner  = rs.align(rs.stream.color)
    pipeline.start(cfg)
    print(f"Stabilizing... please hold still for {wait_time} seconds")
    time.sleep(wait_time)
    # grab one good frame
    while True:
        frames      = pipeline.wait_for_frames()
        aligned     = aligner.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if depth_frame and color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            break
    # get intrinsics for deprojection
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    pipeline.stop()
    h, w, _ = color_image.shape
    # — MediaPipe pose on color —
    rgb     = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if not results.pose_landmarks:
        raise RuntimeError("Failed to detect torso landmarks")
    # corner pixel coords
    coords = {}
    for lm in TORSO_LANDMARKS:
        p = results.pose_landmarks.landmark[lm]
        coords[lm.name] = (p.x * w, p.y * h)
    # full grid
    grid = interpolate_grid(
        coords["LEFT_SHOULDER"],
        coords["RIGHT_SHOULDER"],
        coords["LEFT_HIP"],
        coords["RIGHT_HIP"],
        n=num_columns,
        m=num_rows
    )
    # annotate frame
    frame = color_image.copy()
    for (x, y) in coords.values():
        cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)
    for row in grid:
        for (x, y) in row:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
    # select subset, deproject, label
    real_points = []
    orientations = []
    idx = 0
    
    for ri in [1, 4, 8, 10]:
        if ri == 4 or ri == 10:
            ci_list = [7, 5, 3, 1]
        else: 
            ci_list = [1, 3, 5, 7]
        for ci in ci_list:
            xi, yi, transformed_point = point_transform(depth_frame, depth_intrin, w, h, grid, ri, ci)
            real_points.append(transformed_point)
            cv2.circle(frame, (xi, yi), 6, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(idx),
                (xi - 10, yi),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            ## Add in the points for the plane
            if (ci == 1 and (ri == 4 or ri == 8)): ## take care of edge cases on the right
                _, _, plane_top_left = point_transform(depth_frame, depth_intrin, w, h, grid, ri - 2, ci)
                _, _, plane_top_right = point_transform(depth_frame, depth_intrin, w, h, grid, ri - 1, ci + 1)
                roll, pitch, yaw = plane_orientation(plane_top_left, plane_top_right, transformed_point)
                xi_tl, yi_tl, plane_top_left = point_transform(
                depth_frame, depth_intrin, w, h, grid, ri - 2, ci
                )
                # top‐right:
                xi_tr, yi_tr, plane_top_right = point_transform(
                    depth_frame, depth_intrin, w, h, grid, ri - 1, ci + 1
                )
                # bottom‐mid:
                xi_bm, yi_bm, plane_bottom = point_transform(
                    depth_frame, depth_intrin, w, h, grid, ri, ci
                )
            else: 
                _, _, plane_top_left = point_transform(depth_frame, depth_intrin, w, h, grid, ri - 1, ci - 1)
                _, _, plane_top_right = point_transform(depth_frame, depth_intrin, w, h, grid, ri - 1, ci + 1)
                roll, pitch, yaw = plane_orientation(plane_top_left, plane_top_right, transformed_point)

                xi_tl, yi_tl, plane_top_left = point_transform(
                    depth_frame, depth_intrin, w, h, grid, ri - 1, ci - 1
                )
                # top‐right:
                xi_tr, yi_tr, plane_top_right = point_transform(
                    depth_frame, depth_intrin, w, h, grid, ri - 1, ci + 1
                )
                # bottom‐mid:
                xi_bm, yi_bm, plane_bottom = point_transform(
                    depth_frame, depth_intrin, w, h, grid, ri, ci
                )

            # draw small circles (e.g. in yellow) at those three pixels:
            cv2.circle(frame, (xi_tl, yi_tl), 5, (0, 0, 0), -1)  # top‐right
            cv2.circle(frame, (xi_tr, yi_tr), 5, (0, 255, 0), -1)  # top‐left
            cv2.circle(frame, (xi_bm, yi_bm), 5, (0, 255, 255), -1)  # bottom (b, g, r)
            
            idx += 1
            orientations.append((roll, pitch, yaw))

    # print them immediately
    print("\n--- TRANSFORMED 3D POINTS (X, Y, Z) ---")
    for i, (X, Y, Z) in enumerate(real_points, start=1):
        print(f"{i}: ({X:.3f}, {Y:.3f}, {Z:.3f})")
    print("\n--- PLANE ORIENTATIONS (radians) ---")
    for i, (r, p, y) in enumerate(orientations, start=1):
        print(f"{i}: roll={r:.3f}, pitch={p:.3f}, yaw={y:.3f}")
    # save annotated image
    img_path = "torso_grid_labeled.png"
    cv2.imwrite(img_path, frame)
    print(f"[get_torso] :frame_with_picture: Saved annotated image to '{img_path}'")
    
    return real_points, orientations # THIS IS THE REAL POINT and ORIENTATIONS


def transform_points(points):
    """
    Rotate each (X,Y,Z) by:
      1) 90° about X
      2) 180° about Z
    """
    θ1 = np.pi/2
    Rx1 = np.array([
        [1,            0,             0],
        [0, np.cos(θ1), -np.sin(θ1)],
        [0, np.sin(θ1),  np.cos(θ1)]
    ])
    θ2 = np.pi
    Rx2 = np.array([
        [np.cos(θ2), -np.sin(θ2),           0],
        [np.sin(θ2),  np.cos(θ2), 0],
        [0, 0, 1]
    ])
    R =  Rx1 @ Rx2
    """
    add translation from camera to robot frame
    """
    return [tuple((R @ np.array(p)).tolist()) for p in points]


def publish_all_xyz_as_list(points, host="localhost", port=6379, db=0):
    if not points:
        print("[publish] :warning: no points to publish")
        return
    try:
        r = redis.Redis(host=host, port=port, db=db, socket_timeout=2)
        # clear old list
        r.delete(SCRUB_POINTS_KEY)
        for i, (X, Y, Z) in enumerate(points, start=1):
            val = f"{X:.6f} {Y:.6f} {Z:.6f}"
            print(f"[publish] :arrow_forward: pushing {SCRUB_POINTS_KEY}[{i}] = '{val}'")
            r.rpush(SCRUB_POINTS_KEY, val)
    except Exception as e:
        print(f"[publish] :x: ERROR: {e}")


if __name__ == "__main__":
    # 1) Capture raw torso points
    true_points, orientations = get_torso_grid_with_depth()
   
    # 3) Save **all** rotated points to file
    txt_path = "torso_points_transformed.txt"
    with open(txt_path, "w") as f:
        for i, (X, Y, Z) in enumerate(true_points, start=1):
            f.write(f"{i}: {X:.6f} {Y:.6f} {Z:.6f}\n")
    print(f"[main] :floppy_disk: Saved all transformed points to '{txt_path}'")

    # 4) Push **all** points into Redis list
    #publish_all_xyz_as_list(true_points)
    publish_all_as_one(true_points=true_points, orientations=orientations)
