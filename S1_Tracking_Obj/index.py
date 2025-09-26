import cv2

def callback(value):
    pass

# tạo 1 hàm có giá trị truyền vào là một khoảng / mảng
def setup_trackbacks(range_filter):
    cv2.namedWindow("Trackbars")

    for i in ["MIN","MAX"]:
        v = 0 if i == "MIN" else 255

        for j in range_filter :
            cv2.createTrackbar("%s_%s" % (j,i), "Trackbars", v, 255, callback)

def get_trackback_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)
    
    return values

def main():
    range_filter = "HSV"

    cap = cv2.VideoCapture(0)
    setup_trackbacks(range_filter)

    while(True):

        _, frame = cap.read()

        frame_to_thresh = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackback_values(range_filter)

        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        cv2.imshow("frame", frame)
        cv2.imshow("thres", thresh)

        if cv2.waitKey(1) & 0xFF is ord('q'):
            break

if __name__ == '__main__':
    main()