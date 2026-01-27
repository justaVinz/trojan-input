def apply_config(args, cfg):
    args.dataset = cfg["dataset"]["name"]
    args.model = cfg["model"]["name"]

    args.set_sizes = cfg["dataset"]["set_sizes"]
    args.poisoning_rates = cfg["dataset"]["poisoning_rates"]

    args.bit_sequences = cfg["triggers"]["bit_sequences"]
    args.simple_triggers = cfg["triggers"]["simple_triggers"]

    args.methods = cfg["methods"]

    args.learning_rate = cfg["training"]["learning_rate"]
    args.weight_decay = cfg["training"]["weight_decay"]
    args.num_epochs = cfg["training"]["epochs"]

    return args
