import sys
import torch


def main():
    state = torch.load(sys.argv[1], map_location="cpu")
    shared = {}
    state_dict = {}

    for key in state["model"]:
        if "shared" in key:
            new_key = key.split(".")[:-2] + key.split(".")[-1:]
            new_key = ".".join(new_key)
            shared[new_key] = state["model"][key]
            state["model"][key] = None
            print("%s -> %s" % (key, new_key))

    for key in state["model"]:
        new_key = key.split(".")[:-3] + key.split(".")[-1:]
        new_key = ".".join(new_key)

        if new_key in shared:
            state["model"][key].add_(shared[new_key])
            print("merge key %s" % key)

        if state["model"][key] is not None:
            state_dict[key] = state["model"][key]

    state["model"] = state_dict

    torch.save(state, sys.argv[2])


if __name__ == "__main__":
    main()
