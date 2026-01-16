from ultralytics import YOLO

def main():
    print("CHOO CHOO")
    model = YOLO("inputModels/yolo26l.pt")
    model.train(
        project='./outputModels',
        data="dataset/data.yaml",
        imgsz=640,
        batch=16,
        epochs=1,
        workers=6,
        device=0
    )

if __name__ == "__main__":
    main()
