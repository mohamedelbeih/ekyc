import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from rdi_liveness_detection import *


parser = argparse.ArgumentParser()
parser.add_argument("-I", "--Input", help = "Example: Input/", required = True)
parser.add_argument("-O", "--Output", help = "Example: Output/", required = False, default = "output/")
parser.add_argument("-M", "--Model", help = "Example: models/liveness_detection/v1.0/model", required = True)
parser.add_argument("-EM", "--Exectution_Mode", help = "sequential or concurrent", required = False, default = "sequential")

args = parser.parse_args()

input_path  = args.Input
output_path = args.Output
exectuion_mode = args.Exectution_Mode
model_path = args.Model


def dump_output(result, case):
    output = ""
    output += "Liveness : " + str(result.liveness) + "\n"
    output += "Confidence : " + " ".join(str(result.actions_percentage[i]) for i in range(result.actions_percentage.size()))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    with open(os.path.join(output_path, case + ".txt"), "w") as f:
        f.write(output)


def get_actions(path):
    actions = ActionList()

    with open(path, "r") as f:
        actions_str = f.read().strip().split(', ')

    for action in actions_str:
        if action == "Center":
            actions.append(Action_LOOKINGCENTER)
        elif action == "Right":
            actions.append(Action_LOOKINGRIGHT)
        elif action == "Left":
            actions.append(Action_LOOKINGLEFT)
        elif action == "Up":
            actions.append(Action_LOOKINGUP)
        elif action == "Down":
            actions.append(Action_LOOKINGDOWN)
        elif action == "Smiling":
            actions.append(Action_SMILING)
        elif action == "Blinking":
            actions.append(Action_BLINKING)

    return actions


def run(case):
    video_name = os.path.splitext(case)[0]
    video_path  = os.path.join(input_path, "liveness_scenario", "videos", case)
    action_path = os.path.join(input_path, "liveness_scenario", "actions", video_name + ".txt")

    actions = get_actions(action_path)
    #output1 = liveness_check_from_path(video_path, model_path, actions, TimeSlotList())
    #dump_output(output1.result, video_name)

    # read video as python bytes
    buffer = []
    with open(video_path, "rb") as file:
        while True:
            byte = file.read(1)
            if not byte:
                break
            buffer.append(byte)

    buff = b''.join(buffer)
    output2 = liveness_check(buff, len(buff), model_path, actions, TimeSlotList())
    dump_output(output2.result, video_name + "-buff")

 
def sequential_execution():
    for dir in os.listdir(os.path.join(input_path, "liveness_scenario", "videos")):
        run(dir)


def parallel_exectution():
    with ThreadPoolExecutor(max_workers=5) as executor:
        for dir in os.listdir(os.path.join(input_path, "liveness_scenario", "videos")):
            executor.submit(run, dir)


def main():
    # check the input folder exists
    if not os.path.exists(input_path):
        print("%s doesn't exist." % (input_path))
        exit(1)

    # initialize the model.
    initialize_model(model_path)
    print("--- Initialization of the model is done ----")
    
    # recognize card and its fields.
    start = time.time()
    if exectuion_mode == "sequential" :
        print("**** RUN SEQUENTIAL TEST ****")
        sequential_execution();
    
    else:
        print("**** RUN PARALLEL TEST ****")
        parallel_exectution()
        pass
    
    exectuion_time = time.time() - start
    print("** %s Time = %f secs"% (exectuion_mode, exectuion_time) )


if __name__ == "__main__":
    main()
