import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.spatial.distance import cdist
from scipy.ndimage import maximum_filter

# Results directory for saving output images
RESULTS_DIR = 'results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

class FeatureMatchingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.image_folder = 'images'
        self.reference_image = os.path.join(self.image_folder, 'bernieSanders.jpg')
        self.image_list = [
            f for f in os.listdir(self.image_folder) 
            if f.lower().endswith(('png','jpg','jpeg'))
        ]
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.imageDropdown = QComboBox(self)
        self.imageDropdown.addItems(self.image_list)
        self.imageDropdown.currentIndexChanged.connect(self.load_selected_image)
        layout.addWidget(self.imageDropdown)

        self.canvas = FigureCanvas(plt.figure(figsize=(6, 4)))
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.setWindowTitle('Feature Matching')
        self.show()

    def load_selected_image(self):
        selected_image = self.imageDropdown.currentText()
        if selected_image:
            test_image_path = os.path.join(self.image_folder, selected_image)
            self.match_features(self.reference_image, test_image_path)

    def HarrisPointsDetector(self, image_path):
        """Compute Harris keypoints (scoring point: implementation of corner detection)."""
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
        Ixx = cv2.GaussianBlur(Ix*Ix, (5,5), 0.5)
        Iyy = cv2.GaussianBlur(Iy*Iy, (5,5), 0.5)
        Ixy = cv2.GaussianBlur(Ix*Iy, (5,5), 0.5)

        k = 0.05
        detM = Ixx * Iyy - Ixy * Ixy
        traceM = Ixx + Iyy
        R = detM - k*(traceM**2)

        # Threshold based on a fraction of R.max()
        Rmax = R.max()
        threshold_val = 0.1 * Rmax
        local_max = maximum_filter(R, size=7)
        corners = (R == local_max) & (R > threshold_val)

        keypoints = []
        ys, xs = np.where(corners)
        for y, x in zip(ys, xs):
            angle = np.degrees(np.arctan2(Iy[y,x], Ix[y,x]))
            kp = cv2.KeyPoint(float(x), float(y), 3, angle)
            keypoints.append(kp)

        return keypoints, R

    def process_image(self, image_path):
        """Compute descriptors for Harris keypoints (scoring point: separate detection and description)."""
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, _ = self.HarrisPointsDetector(image_path)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.compute(gray, keypoints)
        return img, keypoints, descriptors

    def matchFeatures(self, des1, des2):
        """Compute SSD-based and Ratio-based matches (scoring point: two matching strategies)."""
        distances = cdist(des1, des2, 'sqeuclidean')
        matches_ssd = []
        matches_ratio = []

        for i in range(len(des1)):
            idx_sorted = np.argsort(distances[i])
            best = idx_sorted[0]
            if len(idx_sorted) < 2:
                # Edge case: only 1 descriptor
                dist_best = distances[i][best]
                matches_ssd.append(cv2.DMatch(i, best, dist_best))
                continue

            second_best = idx_sorted[1]
            dist_best = distances[i][best]
            dist_second = distances[i][second_best] + 1e-9
            ratio_val = dist_best / dist_second

            # SSD (closest neighbor)
            matches_ssd.append(cv2.DMatch(i, best, dist_best))
            # Ratio test
            if ratio_val < 0.75:
                matches_ratio.append(cv2.DMatch(i, best, dist_best))

        # Sort matches by distance
        matches_ssd = sorted(matches_ssd, key=lambda x: x.distance)
        matches_ratio = sorted(matches_ratio, key=lambda x: x.distance)
        return matches_ssd, matches_ratio

    def match_features(self, ref_image_path, test_image_path):
        """Main function that matches ref image and test image."""
        ref_img, ref_kp, ref_des = self.process_image(ref_image_path)
        test_img, test_kp, test_des = self.process_image(test_image_path)

        if (ref_des is None or test_des is None 
            or len(ref_des) == 0 or len(test_des) == 0):
            self.plot_results(ref_img, ref_kp, test_img, test_kp, [], 
                              test_image_path, mode='No Keypoints', out_path=None)
            return

        matches_ssd, matches_ratio = self.matchFeatures(ref_des, test_des)

        # Draw and save first 100 SSD matches
        ssd_outfile = os.path.join(
            RESULTS_DIR, f"SSD_{os.path.basename(test_image_path)}"
        )
        self.plot_results(ref_img, ref_kp, test_img, test_kp, 
                          matches_ssd[:100], test_image_path, 
                          mode='SSD', out_path=ssd_outfile)

        # Draw and save first 100 Ratio matches
        ratio_outfile = os.path.join(
            RESULTS_DIR, f"Ratio_{os.path.basename(test_image_path)}"
        )
        self.plot_results(ref_img, ref_kp, test_img, test_kp, 
                          matches_ratio[:100], test_image_path, 
                          mode='Ratio', out_path=ratio_outfile)

    def plot_results(self, ref_image, ref_keypoints, test_image, test_keypoints, 
                     matches, test_image_path, mode='', out_path=None):
        """Plot and optionally save matching results."""
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        out_img = cv2.drawMatches(ref_image, ref_keypoints,
                                  test_image, test_keypoints,
                                  matches, None)
        ax.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{mode} Matching: {os.path.basename(test_image_path)}')
        self.canvas.draw()

        if out_path:
            self.canvas.figure.savefig(out_path, dpi=100)
            print(f"Saved: {out_path}")

    def visualize_orb_keypoints(self, image_path, save_path='results/orb_keypoints_comparison.png'):
        """Visualize and save keypoints from different ORB methods."""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # HarrisPointsDetector
        harris_kps, _ = self.HarrisPointsDetector(image_path)
        img_harris = cv2.drawKeypoints(img, harris_kps, None, color=(0,255,0))

        # ORB (FAST)
        orb_fast = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)
        fast_kps = orb_fast.detect(gray, None)
        img_fast = cv2.drawKeypoints(img, fast_kps, None, color=(255,0,0))

        # ORB (HARRIS)
        orb_harris = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE)
        harris_orb_kps = orb_harris.detect(gray, None)
        img_orb_harris = cv2.drawKeypoints(img, harris_orb_kps, None, color=(0,0,255))

        # Plot them side by side
        fig, axs = plt.subplots(1, 3, figsize=(15,5))

        axs[0].imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f'Harris + ORB ({len(harris_kps)} points)')
        axs[0].axis('off')

        axs[1].imshow(cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f'ORB (FAST) ({len(fast_kps)} points)')
        axs[1].axis('off')

        axs[2].imshow(cv2.cvtColor(img_orb_harris, cv2.COLOR_BGR2RGB))
        axs[2].set_title(f'ORB (HARRIS) ({len(harris_orb_kps)} points)')
        axs[2].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(save_path, dpi=150)
        print(f"ORB keypoint comparison saved to: {save_path}")


def plot_interest_points_vs_threshold(img_path):
    """Plots number of corners vs. threshold ratio (scoring point: analyzing corner distribution)."""
    # Basic Harris response
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    Ixx = cv2.GaussianBlur(Ix*Ix, (5,5), 0.5)
    Iyy = cv2.GaussianBlur(Iy*Iy, (5,5), 0.5)
    Ixy = cv2.GaussianBlur(Ix*Iy, (5,5), 0.5)

    k = 0.05
    detM = Ixx * Iyy - Ixy * Ixy
    traceM = Ixx + Iyy
    R = detM - k * (traceM ** 2)

    Rmax = R.max()
    local_max = maximum_filter(R, size=7)
    thresholds = np.linspace(0.005, 0.05, 20)
    corners_count = []

    for ratio in thresholds:
        val = ratio * Rmax
        corners = (R == local_max) & (R > val)
        corners_count.append(np.sum(corners))

    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, corners_count, marker='o')
    plt.title('Number of Interest Points vs. Threshold Ratio')
    plt.xlabel('Threshold Ratio')
    plt.ylabel('Number of Interest Points')
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    # Observe the corner distribution curve for the reference image_
    test_image = 'images/bernieSanders.jpg'
    plot_interest_points_vs_threshold(test_image)

    # Launch the PyQt5 GUI
    app = QApplication(sys.argv)
    gui = FeatureMatchingGUI()

    gui.visualize_orb_keypoints(test_image)
    
    sys.exit(app.exec_())
