import os
import json
import logging
import torch.utils.data as data

from .utils import BASE_DIR, META_DIR

class BRACE_Dataset(data.Dataset):
    """
    Dataset Format:
    {
        'audio_path': 'path/to/audio/file',
        'caption_0': 'caption_0',
        'caption_1': 'caption_1',
        'answer': 0 or 1,
        'references': ['reference_1', 'reference_2', ...]
        'pair_type': pair_type, (for Main type)
        'caption_type': [type_0, type_1] (for Main type)
    }
    """

    def __init__(self, meta_path, meta_type, audio_base_dir, processed=False):
        self.meta_path = meta_path
        self.meta_type = meta_type
        self.audio_base_path = audio_base_dir
        
        self.data = []
        with open(meta_path, 'r', encoding='utf8') as f:
            self.json_data = json.load(f)

        if processed:
            self.data = self.json_data
            return
        
        if self.meta_type == 'Hallu':
            for audio_item in self.json_data:
                audio_path = os.path.join(audio_base_dir, audio_item['file_name'])
                if not os.path.exists(audio_path):
                    logging.error(f"Audio file not found: {audio_path}")
                    continue
                    
                for key, value in audio_item.items():
                    if 'caption' in key: # caption_i, i=1,2,3,4,5
                        caption_0 = value[0]
                        caption_1 = value[1]
                        answer = 0 if value[2] == 'human' else 1
                        references = value[-1]['references']

                        if caption_0 in references:
                            references.remove(caption_0)
                        if caption_1 in references:
                            references.remove(caption_1)

                        self.data.append({
                            'audio_path': audio_path,
                            'caption_0': caption_0,
                            'caption_1': caption_1,
                            'answer': answer,
                            'references': references
                        })
        elif self.meta_type == 'Main':
            for audio_item in self.json_data:
                audio_path = os.path.join(audio_base_dir, audio_item['file_name'])
                if not os.path.exists(audio_path):
                    logging.error(f"Audio file not found: {audio_path}")
                    continue

                references = audio_item.get('references', [])
                for key, value in audio_item.items():
                    if key in ['file_name', 'references']:
                        continue

                    caption_0, caption_1 = value[0], value[1]
                    type_0, type_1 = value[2], value[3]

                    score_list = value[-1]
                    total_score = sum(score_list)
                    answer = 0 if total_score > 0 else (1 if total_score < 0 else "tie")
                    
                    # 复制一份 references，并移除与当前 captions 重复的部分
                    # list(references) 能够避免引用问题
                    pair_references = list(references)
                    if caption_0 in pair_references:
                        pair_references.remove(caption_0)
                    if caption_1 in pair_references:
                        pair_references.remove(caption_1)
                    
                    # 额外记录 score, pair_type, caption_type 便于后续筛选
                    self.data.append({
                        'audio_path': audio_path,
                        'caption_0': caption_0,
                        'caption_1': caption_1,
                        'answer': answer,
                        'references': pair_references,
                        'score': total_score,
                        'pair_type': key, 
                        'caption_type': [type_0, type_1], 
                    })

        else:
            logging.error(f"Invalid BRACE Dataset type: {self.meta_type}")
            raise ValueError(f"Invalid BRACE Dataset type: {self.meta_type}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # TODO: __getitem__ 用于随机访问数据集中的某个样本，使用 dict 更加方便
        # 如果要结合 dataloader 进行 batch 操作，tuple 能够正常 collate_fn，dict 需要自定义
        return self.data[idx]
    
    def get_all_audios(self):
        # NOTE: 如果是 SLIDE evaluate，使用 CLAP encode 时为了加速需要去重
        # 如果是 LLM evaluate，不同的 item 之间没有关联
        audios = [item['audio_path'] for item in self.data]
        audios = list(set(audios))
        return audios

    def get_all_captions(self, with_refs=False):
        captions = []
        for item in self.data:
            captions.append(item['caption_0'])
            captions.append(item['caption_1'])
            if with_refs:
                captions.extend(item['references'])
        captions = list(set(captions))
        return captions
    
    def print_info(self):
        num_samples = len(self.data)
        num_audio = len(self.get_all_audios())
        num_captions = len(self.get_all_captions())
        print(f"Task Name: {os.path.basename(self.meta_path)}")
        print(f"Meta Path: {self.meta_path}")
        print(f"Audio Base Path: {self.audio_base_path}")
        print(f"Dataset Info: {num_samples} samples, {num_audio} unique audio files, {num_captions} unique captions")
        print("Sample data: ")
        print(json.dumps(self.data[0], indent=4, ensure_ascii=False))
        print('-' * 50)


def test_Hallu():
    AudioCaps_dataset = BRACE_Dataset(
        meta_path=os.path.join(META_DIR, 'AudioCaps_Hallu_v2s.json'),
        meta_type='Hallu', 
        audio_base_dir=os.path.join(BASE_DIR, 'data/BRACE/Hallu/AudioCaps/audio')
    )
    AudioCaps_dataset.print_info()

    Clotho_dataset = BRACE_Dataset(
        meta_path=os.path.join(META_DIR, 'Clotho_Hallu_v2s.json'),
        meta_type='Hallu', 
        audio_base_dir=os.path.join(BASE_DIR, 'data/BRACE/Hallu/Clotho/audio')
    )
    Clotho_dataset.print_info()


def test_Main():
    AudioCaps_dataset = BRACE_Dataset(
        meta_path=os.path.join(META_DIR, 'AudioCaps_Main_v2.json'), 
        meta_type='Main', 
        audio_base_dir=os.path.join(BASE_DIR, 'data/BRACE/Main/AudioCaps/audio')
    )
    AudioCaps_dataset.print_info()

    Clotho_dataset = BRACE_Dataset(
        meta_path=os.path.join(META_DIR, 'Clotho_Main_v2.json'),
        meta_type='Main', 
        audio_base_dir=os.path.join(BASE_DIR, 'data/BRACE/Main/Clotho/audio')
    )
    Clotho_dataset.print_info()


if __name__ == "__main__":
    test_Hallu()
    test_Main()