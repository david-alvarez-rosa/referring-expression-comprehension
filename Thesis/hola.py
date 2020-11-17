
class ReferDataset(data.Dataset):
    """Refer Dataset."""
    def __init__(self, ann_file, ref_file, split, transform=None):
        self.refer = REFER(ann_file, ref_file)
        self.ref_ids = self.refer.getRefIds(split=split)
        self.transform = transform

    def __len__(self):
        return len(self.ref_ids)

    def __repr__(self):
        print("hola, representation done!")

    def __getitem__(self, index):
        """Get item."""
        ref_id = self.ref_ids[index]
        ref = self.refer.loadRefs(ref_id)[0]

        # Image.
        img_id = ref['image_id']
        img = self.refer.loadImgs(img_id)[0]
        img_path = ('/home/david/Documents/UPC/Cuatrimestre 9/'
                    'cocoapi/coco/train2014/{}'.format(img['file_name']))
        img = io.imread(img_path)
        if self.transform is not None:
            img = self.transform(img)

        # Sentences.
        sents = ref['sents']

        sample = {'img': img, 'sents': sents, 'target': target}
        return sample
