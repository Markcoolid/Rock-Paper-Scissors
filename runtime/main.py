import cv2 as cv
import random
from ultralytics import YOLO

yolo_model = YOLO("models/model.pt")

def playGame(userPlay, computerPlay):
    if(userPlay == computerPlay):
        return "TIE"
    if userPlay == 0 and computerPlay == 1:
        return "PLAYER WINS"
    if userPlay == 1 and computerPlay == 2:
        return "PLAYER WINS"
    if userPlay == 2 and computerPlay == 0:
        return "PLAYER WINS"
    return "COMPUTER WINS"


def get_color(cls: int):
    #Paper color
    if cls == 0:
        return (255,255,255)
    #Rock color
    if cls == 1:
        return (50,100,100)
    #Scissor color
    return (100,0,0)

def center_text(img, text, heightOffset, font, font_scale, font_thickness, color):
    img_height, img_width = img.shape[:2]

    text_size, baseline = cv.getTextSize(text, font, font_scale, font_thickness)
    text_width, text_height = text_size


    text_x = (img_width - text_width) // 2
    text_y = (img_height + text_height) // 2 - heightOffset

    cv.putText(img, text, (text_x, text_y), font, font_scale, color, font_thickness, cv.LINE_AA)
    return img

def main():
    print("Initializing Rock Paper Scissors")
    cam = cv.VideoCapture(0)

    while not (cv.waitKey(1) & 0xFF == ord(' ')):
        ret, frame = cam.read()
        if not ret:
            print("Unable To Read Frame")
            cv.destroyAllWindows()
            cam.release()
            exit()
        
        cv.rectangle(
            frame,
            (0,0),
            (640, 480),
            (0,0,0),
            -1
        )

        frame = center_text(frame, "Rock Paper Scissors", 100, cv.FONT_HERSHEY_SIMPLEX, 1, 1, (255,255,255))

        frame = center_text(frame, "Press Space To Start", 0, cv.FONT_HERSHEY_SIMPLEX, 1, 1, (255,255,255))

        cv.imshow("Rock Paper Scissors", frame)
        
        if(cv.waitKey(1) & 0xFF == ord('q')):
            cv.destroyAllWindows()
            cam.release()
            exit()

    ret, frame = cam.read()

    cv.rectangle(
        frame,
        (0,0),
        (640, 480),
        (0,0,0),
        -1
    )
    frame = center_text(frame, "ROCK", 0, cv.FONT_HERSHEY_SIMPLEX, 2, 3, (255,255,255))
    cv.imshow("Rock Paper Scissors", frame)
    cv.waitKey(1000)

    ret, frame = cam.read()

    cv.rectangle(
        frame,
        (0,0),
        (640, 480),
        (255,255,255),
        -1
    )
    frame = center_text(frame, "PAPER", 0, cv.FONT_HERSHEY_SIMPLEX, 2, 3, (0,0,0))
    cv.imshow("Rock Paper Scissors", frame)
    cv.waitKey(1000)
    ret, frame = cam.read()

    cv.rectangle(
        frame,
        (0,0),
        (640, 480),
        (0,0,0),
        -1
    )
    frame = center_text(frame, "SCISSORS", 0, cv.FONT_HERSHEY_SIMPLEX, 2, 3, (255,255,255))
    cv.imshow("Rock Paper Scissors", frame)
    cv.waitKey(1000)

    userPlay = -1
    while userPlay == -1:
        ret, frame = cam.read()
        results = yolo_model.track(frame, stream=True)
        Renderframe = frame.copy()
        cv.rectangle(
            Renderframe,
            (0,0),
            (640, 480),
            (0,0,0),
            -1
        )
        for result in results:
            for box in result.boxes:
                if box.conf[0] > .6:
                    Renderframe = center_text(Renderframe, result.names[int(box.cls[0])], 0, cv.FONT_HERSHEY_SIMPLEX, 2, 3, (255,255,255))
                    userPlay = int(box.cls[0])
                    break
            if not userPlay == -1:
                break

        cv.imshow("Rock Paper Scissors", Renderframe)
        if(cv.waitKey(1) & 0xFF == ord('q')):
            break

    cv.rectangle(
        frame,
        (0,0),
        (640, 480),
        (0,0,0),
        -1
    )

    computerPlay = random.randint(0, 2)
    frame = center_text(frame, "COMPUTER", 200, cv.FONT_HERSHEY_SIMPLEX, 1, 2, (255,255,255))
    frame = center_text(frame, result.names[computerPlay].upper(), 150, cv.FONT_HERSHEY_SIMPLEX, 2, 3, (255,255,255))
    frame = center_text(frame, "PLAYER", -150, cv.FONT_HERSHEY_SIMPLEX, 1, 2, (255,255,255))
    frame = center_text(frame, result.names[userPlay].upper(), -200, cv.FONT_HERSHEY_SIMPLEX, 2, 3, (255,255,255))
    frame = center_text(frame, playGame(userPlay, computerPlay), 0, cv.FONT_HERSHEY_SIMPLEX, 1, 2, (255,255,255))


    

    cv.imshow("Rock Paper Scissors", frame)
    cv.waitKey(5000)

    cv.destroyAllWindows()
    cam.release()


if __name__ == "__main__":
    main()