
def observe(robot, H, n=N_IMAGES, period=PERIOD):
    robot.grab_image()                            # nastartuje kameru
    images, times = [], []
    t_start = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        images.append(robot.grab_image())
        times.append(0.5 * (t0 + time.perf_counter()))   # absolutní čas (perf_counter)
        if i < n - 1:
            time.sleep(max(0.0, t_start + (i + 1) * period - time.perf_counter()))
    t, centers = detectmarkers(images, np.array(times), H, IDS)
    if len(t) < 4:
        raise RuntimeError(f"Značky nalezeny jen na {len(t)} snímcích")
    model,  = fit_rotation(t, centers)
    return model                                  # predict_board(model, time.perf_counter()) platí v tomto běhu


if name == "main":
    H = np.load("calib/H.npy")
    robot = RobotBosch()
    try:
        robot.initialize()
        robot.set_speed_relative(0.3)
        fix_negative_uhel(robot, True)
        robot.move_to_q(OBS_Q)
        robot.wait_for_motion_stop()

        model = observe(robot, H)
        print(f"střed {np.round(model['rotation_center'], 4)}, R {model['radius']:.4f}, "
              f"omega {model['omega_deg_s']:.2f} °/s, pokryto {model['angle_span_deg']:.0f}°")
        with open("table_model.json", "w") as f:
            json.dump(model, f, indent=2)

        fix_negative_uhel(robot, False)
        robot.soft_home()
    finally:
        robot.close()
