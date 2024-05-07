# a QT app that displays a normal map of a given image

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QFileDialog, QHBoxLayout, QPushButton, QListWidget, QColorDialog, QDialog, QSpinBox, QSlider
from PyQt5.QtGui import QKeyEvent, QPixmap, QImage, QColor, QPainter, QPen
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator


from segment_anything import SamPredictor, sam_model_registry

pred = None

def load_model():
    global pred
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    pred = SamPredictor(sam)
    
    
def predict(pts, labels):
    global pred
    masks, _, _ = pred.predict(point_coords=pts, point_labels=labels, multimask_output=False)
    return masks
    


normal_length = 128

img_arr = None

def r(x):
    return int(x)

def cap_vec(x, y):
    length = np.sqrt(x**2 + y**2)
    if length > normal_length:
        x = (x / length) * normal_length
        y = (y / length) * normal_length
    return r(x), r(y)

def normal_color(x, y):
    z = np.sqrt(normal_length**2 - x**2 - y**2)
    # Normalize to [0, 255]
    x = (x / normal_length) * 127
    y = (-y / normal_length) * 127
    z = (z / normal_length) * 127
    return QColor(r(x + 128), r(y + 128), r(z + 128))

def normal_color_v(x, y):
    z = np.sqrt(normal_length**2 - x**2 - y**2)
    # Normalize to [0, 255]
    x = (x / normal_length) * 127
    y = (-y / normal_length) * 127
    z = (z / normal_length) * 127
    return np.array([r(x + 128), r(y + 128), r(z + 128)])

def normal_to_color(x, y, z):
    # Normalize to [0, 255]
    x = (x / normal_length) * 127
    y = (-y / normal_length) * 127
    z = (z / normal_length) * 127
    # cap
    return QColor(r(x + 128), r(y + 128), r(z + 128))

def open_image(image_path):
    # Load image
    global img_arr
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = np.array(img)
    img_arr = img
    # Convert to QImage
    img = img.astype(np.uint8)
    img = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)

    return img

def multiply_image(seg_temp: np.ndarray) -> QImage:
    global img_arr
    mask = seg_temp.copy()
    mask = (mask + 1) / 2
    #duplicate the mask to 3 channels (x, y, 3)
    mask = np.stack([mask, mask, mask], axis=-1)
    img_temp = img_arr * mask
    
    img = img_temp.astype(np.uint8)
    img = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
    return img

def raw_normal_map(width, height):
    # set every pixel to 128, 128, 255
    
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = [128, 128, 255]
    img = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
    return img

def raw_normal_map_with_mask(width, height, mask):
    # set every pixel to 128, 128, 255
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = [128, 128, 255]
    img = img * mask[:, :, np.newaxis]
    img = img.astype(np.uint8)
    return img

def raw_map(width, height):
    # set every pixel to 128, 128, 255
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = [128, 128, 255]
    img = img.astype(np.uint8)
    return img

def label_to_img_coords(label, img, x, y):
    return x/label.width()*img.width(), y/label.height()*img.height()

def nearest_neighbor_interpolatior(points, values):
    def nn_interpolator(indices):
        return np.array([values[np.argmin(np.linalg.norm(points - index, axis=1))] for index in indices])
    
    return nn_interpolator

def itpl_normal(mask, lines, use_mask=True):
    if(len(lines) == 0 or len(lines) == 2):
        if use_mask:
            return raw_normal_map_with_mask(img_arr.shape[1], img_arr.shape[0], mask)
        else:
            return raw_normal_map(img_arr.shape[1], img_arr.shape[0])
        
    if(len(lines) == 1):
        _, _, x, y = lines[0]
        color = normal_color_v(*cap_vec(x, y))
        # every pixel is the same color
        img = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype=np.uint8)
        img[:, :] = color
        img = img.astype(np.uint8)
        return img * mask[:, :, np.newaxis] if use_mask else img
    
    points = np.array([[x, y] for y, x, _, _ in lines])
    values = np.array([[x, y] for _, _, x, y in lines])
    rbfi = RBFInterpolator(points, values, kernel='thin_plate_spline', epsilon=2)
    
    #rbfi = nearest_neighbor_interpolatior(points, values)
    
    print("points:", points)
    print("values:", values)
    
    # same size as img_arr
    normal = np.zeros((img_arr.shape[0], img_arr.shape[1], 2))
    
    # compute normal for each pixel
    if use_mask:
        print("maskshape:",mask.shape)
        mask_indices = np.argwhere(mask)
        normal[mask_indices[:, 0], mask_indices[:, 1]] = rbfi(mask_indices)
    else:
        mask_indices = np.argwhere(np.ones(img_arr.shape[:2]))
        normal[mask_indices[:, 0], mask_indices[:, 1]] = rbfi(mask_indices)
    # convert normal vector to colors (need to invert y component)
    
    
    # print("normal", normal[points[:, 0], points[:, 1]])
    
    normal = np.array([normal_color_v(*cap_vec(x, y)) for x, y in normal.reshape(img_arr.shape[0]*img_arr.shape[1],2)]).reshape(img_arr.shape)
    # clip to [0, 255]
    normal = np.clip(normal, 0, 255)
    
    print(normal.shape)
    
    # mask the normal map
    if use_mask:
        normal = normal * mask[:, :, np.newaxis]
    normal = normal.astype(np.uint8)
    return normal

class LightDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Light Color | Radius Selection")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # Remove "?" help button

        self.layout = QVBoxLayout(self)

        # Color selection components
        self.redSlider = self.create_color_slider()
        self.greenSlider = self.create_color_slider()
        self.blueSlider = self.create_color_slider()

        # Setup layout for color sliders
        colorLayout = QHBoxLayout()
        colorLayout.addWidget(self.redSlider)
        colorLayout.addWidget(self.greenSlider)
        colorLayout.addWidget(self.blueSlider)
        self.layout.addLayout(colorLayout)

        # Label to display the selected color
        self.colorLabel = QLabel("Selected Color: #000000")
        self.colorLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.colorLabel)

        # SpinBox for number input
        self.spinBox = QSpinBox(self)
        self.spinBox.setRange(0, 100)
        self.layout.addWidget(self.spinBox)

        # Confirm button
        self.confirmButton = QPushButton("Ok", self)
        self.confirmButton.clicked.connect(self.accept)  # Connect the button to accept the dialog
        self.layout.addWidget(self.confirmButton)

        # Connect sliders to update color
        self.redSlider.valueChanged.connect(self.update_color)
        self.greenSlider.valueChanged.connect(self.update_color)
        self.blueSlider.valueChanged.connect(self.update_color)

    def create_color_slider(self):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 255)
        return slider

    def update_color(self):
        r = self.redSlider.value()
        g = self.greenSlider.value()
        b = self.blueSlider.value()
        self.colorLabel.setText(f"Selected Color: #{r:02x}{g:02x}{b:02x}")
        self.colorLabel.setStyleSheet(f"QLabel {{ background-color: #{r:02x}{g:02x}{b:02x}; color : #ffffff; }}")


def runDialog():
    dialog = LightDialog()
    if dialog.exec_():
        r=dialog.redSlider.value()
        g=dialog.greenSlider.value()
        b=dialog.blueSlider.value()
        print("Selected Color:", QColor(r, g, b))
        print("Selected Radius:", dialog.spinBox.value())
        return np.array([r,g,b]), dialog.spinBox.value()
    else:
        return None
    
lines = [[]]
seg_points = []
seg_label = []
seg_region = []
normals = []

lights = []

def render_point(x, y, normal_map, label, img):
    global img_arr
    color = img_arr[x,y]
    for (lx, ly, light_color, radius) in lights:
        lx, ly = label_to_img_coords(label, img, lx, ly)
        dist = np.sqrt((x-lx)**2 + (y-ly)**2)
        vec = np.array([x-lx, y-ly])
        if(np.linalg.norm(vec) == 0):
            continue
        vec = vec / np.linalg.norm(vec)
        intensity = (radius - dist)/radius
        if intensity > 0:
            normal_vector = (normal_map[x,y,0]-128)/128, (normal_map[x,y,1]-128)/128
            dot = np.dot(normal_vector, vec)
            dot = dot*intensity
            color = np.clip(light_color * (dot) + color, 0, 255)
            
    return color



def combine_normals():
    # combine all the normal maps, maps at later layers will overwrite earlier layers
    global normals
    normal = np.zeros(img_arr.shape, dtype=np.uint8)
    
    for i in range(len(normals)):
        normal = np.where(normals[i] > 0, normals[i], normal)

    return normal

max_width = 600
max_height = 400

class NormalMap(QWidget):
    def __init__(self, image_path):
        super().__init__()

        # Create QLabel for the left image
        self.label_left = QLabel()
        self.label_left.setAlignment(Qt.AlignCenter)

        # Create QLabel for the right image
        self.label_right = QLabel()
        self.label_right.setAlignment(Qt.AlignCenter)

        # Create menu bar
        menu_bar = QHBoxLayout()
        
        self.opMode = 0 # 0 for add normal, 1 for adjust normal, 2 for segment, 4 for light
        
        self.selectIdx = -1
        self.selectedMask = -1

        # Create buttons for menu bar
        button1 = QPushButton("Add Normal")
        button1.setFixedWidth(100)

        button1.mouseReleaseEvent = self.adjFalse
        
        button2 = QPushButton("Adjust Normal")
        button2.setFixedWidth(150)
        button2.mouseReleaseEvent = self.adjTrue
        
        button3 = QPushButton("Seg Anchors")
        button3.mouseReleaseEvent = self.adjSeg
        button3.setFixedWidth(150)
        
        button5 = QPushButton("Segment")
        button5.mouseReleaseEvent = self.seg
        button5.setFixedWidth(100)
        
        button6 = QPushButton("Update Normal")
        button6.mouseReleaseEvent = self.updateNormal
        button6.setFixedWidth(150)
        
        button7 = QPushButton("Combine Normals")
        button7.mouseReleaseEvent = self.showCombined
        button7.setFixedWidth(150)
        
        button8 = QPushButton("Lights")
        button8.mouseReleaseEvent = self.adjLight
        button8.setFixedWidth(100)
        
        button9 = QPushButton("Render")
        button9.mouseReleaseEvent = self.render
        button9.setFixedWidth(100)
        
        

        # Add buttons to menu bar
        menu_bar.addWidget(button1)
        menu_bar.addWidget(button2)
        menu_bar.addWidget(button3)

        menu_bar.addWidget(button5)
        menu_bar.addWidget(button6)
        menu_bar.addWidget(button7)
        menu_bar.addWidget(button8)
        menu_bar.addWidget(button9)
        
        menu_bar.setAlignment(Qt.AlignTop)
        # Set layout
        layout = QVBoxLayout()
        layout.addLayout(menu_bar)
        
        hor = QHBoxLayout() 
        hor.addWidget(self.label_left)
        hor.addWidget(self.label_right)
        
        # list of masks for segmenting
        self.list = QListWidget()
        hor.addWidget(self.list)
        
        self.list.addItem("No Masks")
        # whenever a new list item is selected, update the right image with the mask
        self.list.currentRowChanged.connect(self.updateSelectedMask)
        self.list.setMaximumWidth(150)
        
        layout.addLayout(hor)
        self.setLayout(layout)

        # Set window title
        self.setWindowTitle('Normal Map')

        # Set window size
        self.resize(1200, 600)

        # Set images
        self.original = open_image(image_path)
    
        
        self.img_left = self.original.scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio)
        img_right = raw_normal_map(self.original.width(), self.original.height())
        img_right = img_right.scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio)
            

        self.label_left.setPixmap(QPixmap.fromImage(self.img_left))
        self.label_right.setPixmap(QPixmap.fromImage(img_right))
        
        #make the label the same size as the image
        self.label_left.setFixedSize(self.img_left.width(), self.img_left.height())
        self.label_right.setFixedSize(img_right.width(), img_right.height())
    
        
        # click and drag mouse to draw a line on the image (always straight line, no curves)
        self.label_left.mousePressEvent = self.mousePressEvent
        self.label_left.mouseMoveEvent = self.mouseMoveEvent
        self.label_left.mouseReleaseEvent = self.mouseReleaseEvent
        
        normals.append(raw_map(self.original.width(), self.original.height()))
        
    def showCombined(self, event):
        combined_normal = combine_normals()
        self.label_right.setPixmap(QPixmap.fromImage(QImage(combined_normal.data, combined_normal.shape[1], combined_normal.shape[0], combined_normal.shape[1] * 3, QImage.Format_RGB888)).scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio))
        
    def updateSelectedMask(self):
        selMask = self.list.currentRow() - 1
        if(selMask != self.selectedMask):
            self.selectedMask = selMask
            if(selMask == -1):
                self.img_left = self.original.scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio)
                self.label_left.setPixmap(QPixmap.fromImage(self.img_left))
            else:
                mask = seg_region[selMask]
                self.img_left = multiply_image(mask).scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio)
                self.label_left.setPixmap(QPixmap.fromImage(self.img_left))
            
            self.opMode = 0
            self.drawLines()
            
            normal = normals[self.selectedMask + 1]
            self.label_right.setPixmap(QPixmap.fromImage(QImage(normal.data, normal.shape[1], normal.shape[0], normal.shape[1] * 3, QImage.Format_RGB888)).scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio))
            
    def updateNormal(self, event):
        #preprocess lines
        img_lines = lines[self.selectedMask+1].copy()
        for i in range(len(img_lines)):
            x, y = label_to_img_coords(self.label_left, self.original, img_lines[i][0], img_lines[i][1])
            a, b = img_lines[i][2]-img_lines[i][0], img_lines[i][3]-img_lines[i][1]
            img_lines[i] = (x, y, a, b)
            
        
        if(self.selectedMask == -1):
            normal = itpl_normal(None, img_lines, use_mask=False)
        else:
            normal = itpl_normal(seg_region[self.selectedMask], img_lines, use_mask=(self.selectedMask != -1))
        normals[self.selectedMask + 1] = normal
        self.label_right.setPixmap(QPixmap.fromImage(QImage(normal.data, normal.shape[1], normal.shape[0], normal.shape[1] * 3, QImage.Format_RGB888)).scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio))
        
        
    def adjFalse(self, event):
        self.opMode = 0
        
    def adjTrue(self, event):
        self.opMode = 1
        
    def adjSeg(self, event):
        self.opMode = 2
        
    def adjLight(self, event):
        self.opMode = 4
        self.drawLights()
        
    def seg(self,event):
        global seg_region, seg_points
        if(len(seg_points) == 0):
            return
        for i in range(len(seg_points)):
            seg_points[i] = label_to_img_coords(self.label_left, self.original, seg_points[i][0], seg_points[i][1])
        seg_temp = predict(np.array(seg_points),np.array(seg_label))
        seg_points.clear()
        seg_label.clear()
        
        #convert seg_region to a mask, seg_region is of shape (1, img_height, img_width) and True/False
        seg_temp = seg_temp[0]
        seg_temp = seg_temp.astype(np.uint8)
        
        seg_region.append(seg_temp)
        lines.append([])
        normals.append(raw_normal_map_with_mask(self.original.width(), self.original.height(), seg_temp))
        
        self.list.addItem("Mask " + str(len(seg_region)))
        self.list.setCurrentRow(len(seg_region))
        
        self.segment = False
        
    
    def mousePressEvent(self, event):
        
        l_temp = lines[self.selectedMask+1]
        
        if self.opMode == 2:
            return
        
        elif self.opMode == 4:
            # TODO: add light
            return
        
        elif self.opMode == 1:
            minDist = 1000000
            self.selectIdx = -1
            l_temp
            for i in range(len(l_temp)):
                x1, y1, x2, y2 = l_temp[i]
                p1dist = (x1-event.x())**2 + (y1-event.y())**2
                p2dist = (x2-event.x())**2 + (y2-event.y())**2
                if p1dist < minDist:
                    minDist = p1dist
                    self.selectIdx = i * 2
                if p2dist < minDist:
                    minDist = p2dist
                    self.selectIdx = i * 2 + 1
                
        elif self.opMode == 0:
            l_temp.append((event.x(), event.y(), event.x(), event.y()))
            
        self.drawLines()
        
    def mouseMoveEvent(self, event):
        if self.opMode == 2 or self.opMode == 4:
            return
        
        l_temp = lines[self.selectedMask+1]
        
        if(self.opMode == 1):
            if(self.selectIdx != -1):
                idx = self.selectIdx//2
                if self.selectIdx % 2 == 0: 
                    offset = (event.x()-l_temp[idx][0], event.y()-l_temp[idx][1])
                    l_temp[idx] = (event.x(), event.y(), l_temp[idx][2]+offset[0], l_temp[idx][3]+offset[1])
                else:
                    v = cap_vec(event.x()-l_temp[idx][0], event.y()-l_temp[idx][1])
                    l_temp[idx] = (l_temp[idx][0], l_temp[idx][1], l_temp[idx][0]+v[0], l_temp[idx][1]+v[1])
                self.drawLines()
        else:
            v = cap_vec(event.x()-l_temp[-1][0], event.y()-l_temp[-1][1])
            l_temp[-1] = (l_temp[-1][0], l_temp[-1][1], l_temp[-1][0]+v[0], l_temp[-1][1]+v[1])
            self.drawLines()
        
    def mouseReleaseEvent(self, event):
        print(event.x(), event.y())
        print(label_to_img_coords(self.label_left, self.original, event.x(), event.y()))
        
        if(self.opMode == 4):
            x = runDialog()
            if(x != None):
                lights.append((event.x(), event.y(), x[0], x[1]))
            self.drawLights()
        
        if(self.opMode == 2):
            seg_points.append((event.x(), event.y()))
            if(event.button() == Qt.LeftButton):
                seg_label.append(1)
            else:
                seg_label.append(0)
            self.drawSeg()
            
    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        # if the key was d
        if(a0.key() == Qt.Key_D) and self.opMode == 0 or self.opMode == 1:
            l_temp = lines[self.selectedMask+1]
            minDist = 1000000
            self.selectIdx = -1
            # get cursor position
            mousePos = self.label_left.mapFromGlobal(self.cursor().pos())
            for i in range(len(l_temp)):
                x1, y1, x2, y2 = l_temp[i]
                p1dist = (x1-mousePos.x())**2 + (y1-mousePos.y())**2
                p2dist = (x2-mousePos.x())**2 + (y2-mousePos.y())**2
                if p1dist < minDist:
                    minDist = p1dist
                    self.selectIdx = i * 2
                if p2dist < minDist:
                    minDist = p2dist
                    self.selectIdx = i * 2 + 1  
            # remove the line
            if self.selectIdx != -1:
                l_temp.pop(self.selectIdx//2)
            
            self.selectIdx = -1
            self.drawLines()
        
        elif(a0.key() == Qt.Key_D) and self.opMode == 2:
            # find closest anchor
            minDist = 1000000
            selAnchor = -1
            mousePos = self.label_left.mapFromGlobal(self.cursor().pos())
            for i in range(len(seg_points)):
                x, y = seg_points[i]
                dist = (x-mousePos.x())**2 + (y-mousePos.y())**2
                if dist < minDist:
                    minDist = dist
                    selAnchor = i
            if selAnchor != -1:
                seg_points.pop(selAnchor)
                seg_label.pop(selAnchor)
                self.drawSeg()
                
        elif (a0.key() == Qt.Key_D) and self.opMode == 4:
            minDist = 1000000
            selLight = -1
            mousePos = self.label_left.mapFromGlobal(self.cursor().pos())
            for i in range(len(lights)):
                x, y, _, _ = lights[i]
                dist = (x-mousePos.x())**2 + (y-mousePos.y())**2
                if dist < minDist:
                    minDist = dist
                    selLight = i
            if selLight != -1:
                lights.pop(selLight)
                self.drawLights()
                
        
    def drawLines(self):
        tempMap = QPixmap.fromImage(self.img_left)
        painter = QPainter(tempMap)
        pen = QPen(Qt.red,2)
        painter.setPen(pen)
        l_temp = lines[self.selectedMask+1]
        for line in l_temp:
            pen.setColor(normal_color(line[2]-line[0], line[3]-line[1]))
            painter.setPen(pen)
            painter.drawLine(line[0], line[1], line[2], line[3])
            painter.drawEllipse(line[2]-3, line[3]-3, 6, 6)
        painter.end()
        self.label_left.setPixmap(tempMap)
        
    def drawSeg(self):
        tempMap = QPixmap.fromImage(self.img_left)
        painter = QPainter(tempMap)
        pen = QPen(Qt.black,2)
        painter.setPen(pen)
        for (point, label) in zip(seg_points, seg_label):
            if(label == 1):
                pen.setColor(Qt.red)
                painter.setPen(pen)
            else:
                pen.setColor(Qt.blue)
                painter.setPen(pen)
            painter.drawEllipse(point[0]-3, point[1]-3, 6, 6)
        painter.end()
        self.label_left.setPixmap(tempMap)
        
    def drawLights(self):
        tempMap = QPixmap.fromImage(self.img_left)
        painter = QPainter(tempMap)
        pen = QPen(Qt.black,2)
        painter.setPen(pen)
        for (x, y, color, radius) in lights:
            pen.setColor(QColor(*color))
            painter.setPen(pen)
            # draw light center
            painter.drawEllipse(x-3, y-3, 6, 6)
            # draw light radius
            pen = QPen(Qt.black,2)
            painter.setPen(pen)
            radius = (int)(radius*self.label_left.width()/self.original.width())
            painter.drawEllipse(x-radius, y-radius, radius*2, radius*2)
        painter.end()
        self.label_left.setPixmap(tempMap)
    
    def render(self, event):
        normal = combine_normals()
        tmpImg = img_arr.copy()
        
        light1 = (25,10)

        imagesize = image.shape[0], image.shape[1]
        scaledSize = 4,2.5
        imageOrigin = 0,0

        yellow_light = np.array([255,255,100])

        # iterate over the image and apply the normal map
        for x in range(tmpImg.shape[0]):
            for y in range(tmpImg.shape[1]):
                # vector to light source
                light = np.array(light1) - (np.array([y,x])/imagesize*scaledSize + imageOrigin)
                dist = np.linalg.norm(light)
                intensity = 10/dist
                # normalize the vector
                light = light / np.linalg.norm(light)
    
                # normal vector
                normal_vector = (normal[x][y][0]-128)/128, (normal[x][y][1]-128)/128
                # dot product
                dot = np.dot(normal_vector, light)
                dot = dot*intensity

                # apply the dot product to the image
                tmpImg[x][y] = np.clip(yellow_light * (dot) + image[x][y]*0.95, 0, 255)
        

        # for x in range(img_arr.shape[0]):
        #     for y in range(img_arr.shape[1]):
        #         tmpImg[x, y] = render_point(x, y, combined_normal, self.label_left, self.original)
                
        img = tmpImg.astype(np.uint8)
        self.label_right.setPixmap(QPixmap.fromImage(QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)).scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio))
            
        
        
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    #open file dialog
    filename = QFileDialog.getOpenFileName()
    
    if(filename[0] == ''):
        sys.exit(0)
        
    load_model()
    
    #open the image into np array
    image = Image.open(filename[0])
    image = np.array(image.convert('RGB'))
    
    pred.set_image(image)
    
    window = NormalMap(filename[0])
    window.setFocusPolicy(Qt.StrongFocus)
    window.show()
    sys.exit(app.exec_())