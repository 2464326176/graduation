import cv2
# 采集图像 并规格化
def generate():
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    # camera = cv2.VideoCapture(0)
    img = cv2.imread('yh.JPG')
    if img.ndim !=2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        cv2.imwrite('yh.pgm', f)
        cv2.imshow("camera", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    generate()
