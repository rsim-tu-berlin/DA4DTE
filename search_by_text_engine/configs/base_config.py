class BaseConfig:

    def __init__(self, args):
        super(BaseConfig, self).__init__()

        self.seed = 42  # random seed
        self.cuda_device = "cuda"  # CUDA device to use

        self.dataset = args.dataset.lower()
        self.dataset_json_file = "/data/augmented_{}.json".format(args.dataset.upper())
        self.dataset_json_file = "/data/augmented_{}.json".format(args.dataset.upper())

        if args.dataset == "ucm":
            self.dataset_image_folder_path = "/home/george/Dropbox/UCM_Captions/images"
        if args.dataset == "rsicd":
            self.dataset_image_folder_path = "/home/george/Dropbox/RSICD/images"

        if args.dataset == "vessel_dataset":
            
            self.dataset_image_folder_path = "/vessel_detection_dataset_v3/png/"
        self._print_config()

    def _print_config(self):
        print("Configuration:", self.__class__.__name__)

        for v in self.__dir__():
            if not v.startswith("_"):
                print("\t{0}: {1}".format(v, getattr(self, v)))
        # import ipdb; ipdb.set_trace()
