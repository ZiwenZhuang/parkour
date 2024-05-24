import os
import os.path as osp
import numpy as np
import torch
import json
from functools import partial
from collections import OrderedDict

from a1_real import UnitreeA1Real, resize2d
from rsl_rl import modules

import rospy
from unitree_legged_msgs.msg import Float32MultiArrayStamped
from sensor_msgs.msg import Image
import ros_numpy

import pyrealsense2 as rs

def get_encoder_script(logdir):
    with open(osp.join(logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)

    model_device = torch.device("cuda")

    unitree_real_env = UnitreeA1Real(
        robot_namespace= "a112138",
        cfg= config_dict,
        forward_depth_topic= "", # this env only computes parameters to build the model
        forward_depth_embedding_dims= None,
        model_device= model_device,
    )

    model = getattr(modules, config_dict["runner"]["policy_class_name"])(
        num_actor_obs= unitree_real_env.num_obs,
        num_critic_obs= unitree_real_env.num_privileged_obs,
        num_actions= 12,
        obs_segments= unitree_real_env.obs_segments,
        privileged_obs_segments= unitree_real_env.privileged_obs_segments,
        **config_dict["policy"],
    )
    model_names = [i for i in os.listdir(logdir) if i.startswith("model_")]
    model_names.sort(key= lambda x: int(x.split("_")[-1].split(".")[0]))
    state_dict = torch.load(osp.join(args.logdir, model_names[-1]), map_location= "cpu")
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(model_device)
    model.eval()

    visual_encoder = model.visual_encoder
    script = torch.jit.script(visual_encoder)

    return script, model_device

def get_input_filter(args):
    """ This is the filter different from the simulator, but try to close the gap. """
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)
    image_resolution = config_dict["sensor"]["forward_camera"].get(
        "output_resolution",
        config_dict["sensor"]["forward_camera"]["resolution"],
    )
    depth_range = config_dict["sensor"]["forward_camera"].get(
        "depth_range",
        [0.0, 3.0],
    )
    depth_range = (depth_range[0] * 1000, depth_range[1] * 1000) # [m] -> [mm]
    crop_top, crop_bottom, crop_left, crop_right = args.crop_top, args.crop_bottom, args.crop_left, args.crop_right
    crop_far = args.crop_far * 1000

    def input_filter(depth_image: torch.Tensor,
            crop_top: int,
            crop_bottom: int,
            crop_left: int,
            crop_right: int,
            crop_far: float,
            depth_min: int,
            depth_max: int,
            output_height: int,
            output_width: int,
        ):
        """ depth_image must have shape [1, 1, H, W] """
        depth_image = depth_image[:, :,
            crop_top: -crop_bottom-1,
            crop_left: -crop_right-1,
        ]
        depth_image[depth_image > crop_far] = depth_max
        depth_image = torch.clip(
            depth_image,
            depth_min,
            depth_max,
        ) / (depth_max - depth_min)
        depth_image = resize2d(depth_image, (output_height, output_width))
        return depth_image
    # input_filter = torch.jit.script(input_filter)

    return partial(input_filter,
        crop_top= crop_top,
        crop_bottom= crop_bottom,
        crop_left= crop_left,
        crop_right= crop_right,
        crop_far= crop_far,
        depth_min= depth_range[0],
        depth_max= depth_range[1],
        output_height= image_resolution[0],
        output_width= image_resolution[1],
    ), depth_range

def get_started_pipeline(
        height= 480,
        width= 640,
        fps= 30,
        enable_rgb= False,
    ):
    # By default, rgb is not used.
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    if enable_rgb:
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    profile = pipeline.start(config)

    # build the sensor filter
    hole_filling_filter = rs.hole_filling_filter(2)
    spatial_filter = rs.spatial_filter()
    spatial_filter.set_option(rs.option.filter_magnitude, 5)
    spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
    spatial_filter.set_option(rs.option.filter_smooth_delta, 1)
    spatial_filter.set_option(rs.option.holes_fill, 4)
    temporal_filter = rs.temporal_filter()
    temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
    temporal_filter.set_option(rs.option.filter_smooth_delta, 1)
    # decimation_filter = rs.decimation_filter()
    # decimation_filter.set_option(rs.option.filter_magnitude, 2)

    def filter_func(frame):
        frame = hole_filling_filter.process(frame)
        frame = spatial_filter.process(frame)
        frame = temporal_filter.process(frame)
        # frame = decimation_filter.process(frame)
        return frame

    return pipeline, filter_func

def main(args):
    rospy.init_node("a1_legged_gym_jetson")

    input_filter, depth_range = get_input_filter(args)
    model_script, model_device = get_encoder_script(args.logdir)
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)
    if config_dict.get("sensor", dict()).get("forward_camera", dict()).get("refresh_duration", None) is not None:
        refresh_duration = config_dict["sensor"]["forward_camera"]["refresh_duration"]
        ros_rate = rospy.Rate(1.0 / refresh_duration)
        rospy.loginfo("Using refresh duration {}s".format(refresh_duration))
    else:
        ros_rate = rospy.Rate(args.fps)

    rs_pipeline, rs_filters = get_started_pipeline(
        height= args.height,
        width= args.width,
        fps= args.fps,
        enable_rgb= args.enable_rgb,
    )

    # gyro_pipeline = rs.pipeline()
    # gyro_config = rs.config()
    # gyro_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    # gyro_profile = gyro_pipeline.start(gyro_config)

    embedding_publisher = rospy.Publisher(
        args.namespace + "/visual_embedding",
        Float32MultiArrayStamped,
        queue_size= 1,
    )

    if args.enable_vis:
        depth_image_publisher = rospy.Publisher(
            args.namespace + "/camera/depth/image_rect_raw",
            Image,
            queue_size= 1,
        )
        network_input_publisher = rospy.Publisher(
            args.namespace + "/camera/depth/network_input_raw",
            Image,
            queue_size= 1,
        )
        if args.enable_rgb:
            rgb_image_publisher = rospy.Publisher(
                args.namespace + "/camera/color/image_raw",
                Image,
                queue_size= 1,
            )

    rospy.loginfo("Depth range is clipped to [{}, {}] and normalized".format(depth_range[0], depth_range[1]))
    rospy.loginfo("ROS, model, realsense have been initialized.")
    if args.enable_vis:
        rospy.loginfo("Visualization enabled, sending depth{} images".format(", rgb" if args.enable_rgb else ""))
    try:
        embedding_msg = Float32MultiArrayStamped()
        embedding_msg.header.frame_id = args.namespace + "/camera_depth_optical_frame"
        frame_got = False
        while not rospy.is_shutdown():
            # Wait for the depth image
            frames = rs_pipeline.wait_for_frames(int( \
                config_dict["sensor"]["forward_camera"]["latency_range"][1] \
                 * 1000)) # ms
            embedding_msg.header.stamp = rospy.Time.now()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            if not frame_got:
                frame_got = True
                rospy.loginfo("Realsense frame recieved. Sending embeddings...")
            if args.enable_rgb:
                color_frame = frames.get_color_frame()
                # Use this branch to log the time when image is acquired
                if args.enable_vis and not color_frame is None:
                    color_frame = np.asanyarray(color_frame.get_data())
                    rgb_image_msg = ros_numpy.msgify(Image, color_frame, encoding= "rgb8")
                    rgb_image_msg.header.stamp = rospy.Time.now()
                    rgb_image_msg.header.frame_id = args.namespace + "/camera_color_optical_frame"
                    rgb_image_publisher.publish(rgb_image_msg)

            # Process the depth image and publish
            depth_frame = rs_filters(depth_frame)
            depth_image_ = np.asanyarray(depth_frame.get_data())
            depth_image = torch.from_numpy(depth_image_.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(model_device)
            depth_image = input_filter(depth_image)
            with torch.no_grad():
                depth_embedding = model_script(depth_image).reshape(-1).cpu().numpy()
            embedding_msg.header.seq += 1
            embedding_msg.data = depth_embedding.tolist()
            embedding_publisher.publish(embedding_msg)
            
            # Publish the acquired image if needed
            if args.enable_vis:
                depth_image_msg = ros_numpy.msgify(Image, depth_image_, encoding= "16UC1")
                depth_image_msg.header.stamp = rospy.Time.now()
                depth_image_msg.header.frame_id = args.namespace + "/camera_depth_optical_frame"
                depth_image_publisher.publish(depth_image_msg)
                network_input_np = (\
                    depth_image.detach().cpu().numpy()[0, 0] * (depth_range[1] - depth_range[0]) \
                    + depth_range[0]
                ).astype(np.uint16)
                network_input_msg = ros_numpy.msgify(Image, network_input_np, encoding= "16UC1")
                network_input_msg.header.stamp = rospy.Time.now()
                network_input_msg.header.frame_id = args.namespace + "/camera_depth_optical_frame"
                network_input_publisher.publish(network_input_msg)

            ros_rate.sleep()
    finally:
        rs_pipeline.stop()

if __name__ == "__main__":
    """ This script is designed to load the model and process the realsense image directly
    from realsense SDK without realsense ROS wrapper
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace",
        type= str,
        default= "/a112138",                    
    )
    parser.add_argument("--logdir",
        type= str,
        help= "The log directory of the trained model",
    )
    parser.add_argument("--height",
        type= int,
        default= 240,
        help= "The height of the realsense image",
    )
    parser.add_argument("--width",
        type= int,
        default= 424,
        help= "The width of the realsense image",
    )
    parser.add_argument("--fps",
        type= int,
        default= 30,
        help= "The fps of the realsense image",
    )
    parser.add_argument("--crop_left",
        type= int,
        default= 60,
        help= "num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_right",
        type= int,
        default= 46,
        help= "num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_top",
        type= int,
        default= 0,
        help= "num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_bottom",
        type= int,
        default= 0,
        help= "num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_far",
        type= float,
        default= 3.0,
        help= "asside from the config far limit, make all depth readings larger than this value to be 3.0 in un-normalized network input."
    )
    parser.add_argument("--enable_rgb",
        action= "store_true",
        help= "Whether to enable rgb image",
    )
    parser.add_argument("--enable_vis",
        action= "store_true",
        help= "Whether to publish realsense image",
    )

    args = parser.parse_args()
    main(args)
