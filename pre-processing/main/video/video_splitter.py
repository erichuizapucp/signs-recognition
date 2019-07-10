from processor import Processor


class VideoSplitter(Processor):
    def __init__(self):
        super().__init__()

    def process(self, data):
        super().process(data)
        seqs = data['sequences']
        print(seqs)
