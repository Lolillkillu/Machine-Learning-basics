from ultralytics import YOLO
import argparse

if __name__ == "main":
    parser = argparse.ArgumentParser
    parser.add_argument("-i", "--input", help = "Input file/stream for predicting OR YAML for dataset to train on", required = True)
    parser.add_argument("-m", "--model", help = "File of model to use. If training, select the pretrained model. Default = ../model/model.pt", default = "../model/model.pt")
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
    model.predict(args.input, save = True)
