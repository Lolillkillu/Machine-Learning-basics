from ultralytics import YOLO
import argparse
import os


def print_detected_sign(results):
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = result.names[class_id]
            print(f"Rozpoznano: {class_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help = "Input file/stream for predicting OR YAML for dataset to train on", required = True)
    parser.add_argument("-m", "--model", help = "File of model to use. If training, select the pretrained model. Default = ../model/best.pt", default = "../model/best.pt")
    parser.add_argument("-t", "--train", help = "Train model", default = False)
    args = parser.parse_args()
    
    if args.train:
        model = YOLO("yolo11m.pt")

        if args.input.endswith("data_template.yaml"):
            import yaml, kagglehub, warnings

            path: str = kagglehub.dataset_download('mikoajkoek/traffic-road-object-detection-polish-12k')

            warnings.warn("I am now expecting ../data/data_template.yaml file to exist... ../data/data.yaml will be overwritten!")

            with open("../data/data_template.yaml", mode = "r") as yaml_in, open("../data/data.yaml", mode = "w") as yaml_out:
                yaml_data = yaml.safe_load(yaml_in)
                yaml_data['path'] = path
                yaml.safe_dump(yaml_data, yaml_out)
            
            model.train(data = "../data/data.yaml")
        else:
            model.train(data = args.input)
        
        exit()

    model = YOLO(args.model)

    file_path = args.input
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Plik {file_path} nie istnieje.")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
        print("Przetwarzanie obrazu...")
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        print("Przetwarzanie filmu...")
    else:
        raise ValueError("Nieobsługiwany format pliku. Użyj innego pliku.")

    results = model.predict(file_path, save=True)
    
    print_detected_sign(results)
    
    print("Koniec UwU")

if __name__ == "__main__":
    main()
