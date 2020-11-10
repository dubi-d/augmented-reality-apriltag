#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2
from renderClass import Renderer
from dt_apriltags import Detector

import rospy
import yaml
import sys
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import rospkg


class ARNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ARNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")

        rospack = rospkg.RosPack()
        # Initialize an instance of Renderer giving the model in input.
        self.renderer = Renderer(rospack.get_path('augmented_reality_apriltag') + '/src/models/duckie.obj')

        self.bridge = CvBridge()

        self.at_detector = Detector(searchpath=['apriltags'], families='tag36h11', nthreads=1, quad_decimate=1.0,
                               quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)

        # subscribe to camera stream
        self.sub_camera_img = rospy.Subscriber("camera_node/image/compressed", CompressedImage, self.callback,
                                               queue_size=1)
        # publish modified image
        self.pub_modified_img = rospy.Publisher(f"~image/compressed", CompressedImage,
                                                queue_size=1)

        calibration_data = self.read_yaml_file(f"/data/config/calibrations/camera_intrinsic/{self.veh}.yaml")
        self.K = np.array(calibration_data["camera_matrix"]["data"]).reshape(3, 3)
        self.camera_params = (self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2])
        self.tag_size = 0.065

        self.log("Letsgoooooo")

    def callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg)  # convert to cv2 img
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect april tag and extract its reference frame
        tags = self.at_detector.detect(grayscale_img, estimate_tag_pose=False, camera_params=None, tag_size=None)
        #self.visualize_at_detection(img, tags)

        # determine H (using apriltag library)
        homography_list = [tag.homography for tag in tags]

        # derive P
        p_list = [self.projection_matrix(self.K, H) for H in homography_list]

        # project the model and draw it (using Renderer)
        for P in p_list:
            img = self.renderer.render(img, P)

        # publish modified image
        img_out = self.bridge.cv2_to_compressed_imgmsg(img)
        img_out.header = msg.header
        img_out.format = msg.format
        self.pub_modified_img.publish(img_out)
    
    def projection_matrix(self, K, H):
        """
            Write here the computation for the projection matrix, namely the matrix
            that maps the camera reference frame to the AprilTag reference frame.
        """
        self.log("\n\n -------- projection_matrix --------")
        self.log(f"K: {type(K)}, shape {K.shape}, \n{K}")
        self.log(f"H: {type(H)}, shape {H.shape}, \n{H}")
        R_2d = np.linalg.inv(K).dot(H)  # R_2d = [r1 r2 t]
        self.log(f"R_2d: shape {R_2d.shape}, \n{R_2d}")
        R_2d = R_2d / np.linalg.norm(R_2d[:, 0])
        r1 = R_2d[:, 0]
        self.log(f"r1: shape {r1.shape}, {r1}")
        r2 = R_2d[:, 1]
        self.log(f"r2: shape {r2.shape}, {r2}")
        t = R_2d[:, 2]
        self.log(f"t: shape {t.shape}, {t}")
        r3 = np.cross(r1, r2)
        self.log(f"r3: shape {r3.shape}, {r3}")
        #r1, r2, r3 = self.orthogonalize(r1, r2, r3)
        R_3d = np.column_stack((r1, r2, r3))
        self.log(f"R_3d before: shape {R_3d.shape}, \n{R_3d}")
        W, U, Vt = cv2.SVDecomp(R_3d)
        self.log(f"W: shape {W.shape}, \n{W}")
        self.log(f"U: shape {U.shape}, \n{U}")
        self.log(f"Vt: shape {Vt.shape}, \n{Vt}")
        R_3d = U.dot(Vt)
        R_3d = np.column_stack((R_3d, t))
        self.log(f"R_3d after: shape {R_3d.shape}, \n{R_3d}")
        P = K.dot(R_3d)
        self.log(f"P: shape {P.shape}, \n{P}")
        return P

    # ToDo clean up
    def orthogonalize(self, b1, b2, b3):
        """
        Compute an orthonormal base from a given (non-orthonormal) base using the Gram-Schmidt process.
        """
        e1 = b1 / np.linalg.norm(b1)
        e2 = b2 - np.dot(b2, e1) * e1
        e2 = e2 / np.linalg.norm(e2)
        e3 = b3 - (np.dot(b3, e1) * e1 + np.dot(b3, e2) * e2)
        e3 = e3 / np.linalg.norm(e3)
        return e1, e2, e3

    def read_image(self, msg_image):
        """
            Convert images to OpenCV images
            Args:
                msg_image (:obj:`CompressedImage`) the image from the camera node
            Returns:
                OpenCV image
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []

    def read_yaml_file(self, fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

    @staticmethod
    def visualize_at_detection(img, tags):
        """
        Visualize detected april tags for debugging.
        """
        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(img, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                         (0, 255, 0))

            cv2.putText(img, str(tag.tag_id),
                        org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(0, 0, 255))

    def on_shutdown(self):
        super(ARNode, self).on_shutdown()


if __name__ == '__main__':
    # Initialize the node
    camera_node = ARNode(node_name='augmented_reality_apriltag_node')
    # Keep it spinning to keep the node alive
    rospy.spin()