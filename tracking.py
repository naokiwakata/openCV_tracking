import cv2
import numpy as np
if __name__ == '__main__':
    """
    Tracking手法を選ぶ。適当にコメントアウトして実行する。
    """
    # Boosting
    # tracker = cv2.TrackerBoosting_create()
    # MIL
    # tracker = cv2.TrackerMIL_create()
    # KCF
    tracker = cv2.TrackerKCF_create()
    # TLD #GPUコンパイラのエラーが出ているっぽい
    # tracker = cv2.TrackerTLD_create()
    # MedianFlow
    # tracker = cv2.TrackerMedianFlow_create()
    # GOTURN # モデルが無いよって怒られた
    # https://github.com/opencv/opencv_contrib/issues/941#issuecomment-343384500
    # https://github.com/Auron-X/GOTURN-Example
    # http://cs.stanford.edu/people/davheld/public/GOTURN/trained_model/tracker.caffemodel
    # tracker = cv2.TrackerGOTURN_create()
    cap = cv2.VideoCapture(0)
    kernel = np.ones((5,5), np.uint8)
    fps = int(cap.get(cv2.CAP_PROP_FPS))                    # カメラのFPSを取得
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))              # カメラの横幅を取得
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))             # カメラの縦幅を取得
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
    video = cv2.VideoWriter('video.mp4', fourcc, fps, (w, h))  
    while True:
        ret, frame = cap.read()
        img= frame
        if not ret:
            continue
         #hsvに変換
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)
        #二値化
        ret,frame= cv2.threshold(h, 60, 255, cv2.THRESH_BINARY)
        #オープニング，クロージング
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        #輪郭を検出。
        labels, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #最大の領域検出
        max_contour = max(contours, key=lambda x: cv2.contourArea(x))
        max_contour=np.array(max_contour)
        p1=tuple(np.min(max_contour,axis=0)[0])
        p2=tuple(np.max(max_contour,axis=0)[0])
        cv2.rectangle(img,p1,p2,color=(0, 255, 0),lineType=cv2.LINE_4,shift=0)
        cv2.imshow('camera', img)                            
        if cv2.waitKey(1) & 0xFF == ord('l'):
            bbox = (p1[0],p1[1],p2[0]-p1[0],p2[1]-p1[1])
            ok = tracker.init(frame, bbox)
            cv2.destroyAllWindows()
            break
    while True:
        # VideoCaptureから1フレーム読み込む
        ret, frame = cap.read()
        if not ret:
            k = cv2.waitKey(1)
            if k == 27 :
                break
            continue
        # Start timer
        timer = cv2.getTickCount()
        # トラッカーをアップデートする
        track, bbox = tracker.update(frame)
        # FPSを計算する
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        # 検出した場所に四角を書く
        if track:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
        else :
            # トラッキングが外れたら警告を表示する
            cv2.putText(frame, "Failure", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);
        # FPSを表示する
        cv2.putText(frame, "FPS : " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);
        # 加工済の画像を表示する
        cv2.imshow("Tracking", frame)
        video.write(frame)    
        # キー入力を1ms待って、k が27（ESC）だったらBreakする
        k = cv2.waitKey(1)
        if k == 27 :
            break
# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()