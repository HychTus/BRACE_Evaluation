import os
import json
import logging
import torch.utils.data as data

class BRACE_Dataset(data.Dataset):
    def __init__(self, meta_path, meta_type, audio_base_path, processed=False):
        self.meta_path = meta_path
        self.meta_type = meta_type
        self.audio_base_path = audio_base_path
        
        self.data = []
        with open(meta_path, 'r', encoding='utf8') as f:
            self.json_data = json.load(f)

        if processed:
            self.data = self.json_data
            return
        
        if self.meta_type == 'hallucination':
            for audio_item in self.json_data:
                audio_path = os.path.join(audio_base_path, audio_item['file_name'])
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
            
        elif self.meta_type == 'comparison':
            raise NotImplementedError
        else:
            logging.error(f"Invalid BRACE Dataset type: {self.meta_type}")
            raise ValueError(f"Invalid BRACE Dataset type: {self.meta_type}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # TODO: __getitem__ 用于随机访问数据集中的某个样本，使用 dict 更加方便
        # 如果要结合 dataloader 进行 batch 操作，tuple 能够正常 collate_fn，dict 需要自定义
        return self.data[idx]
    
    def get_all_audio(self):
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
    

if __name__ == "__main__":
    Clotho_dataset = BRACE_Dataset(
        meta_path='/mnt/public/data/lh/chy/evaluation/metadata/Clotho_Hallu_v1.json', 
        meta_type='hallucination', 
        audio_base_path='/mnt/public/data/lh/chy/data/Brace/Hallu/Clotho/audio'
    )
    print(len(Clotho_dataset))
    print(Clotho_dataset[0])

    AudioCaps_dataset = BRACE_Dataset(
        meta_path='/mnt/public/data/lh/chy/evaluation/metadata/AudioCaps_Hallu_v1.json', 
        meta_type='hallucination', 
        audio_base_path='/mnt/public/data/lh/chy/data/Brace/Hallu/AudioCaps/audio'
    )
    print(len(AudioCaps_dataset))
    print(AudioCaps_dataset[0])

    audios = Clotho_dataset.get_all_audio()
    captions = Clotho_dataset.get_all_captions()
    print(len(audios), len(captions))
    print(audios[:5], captions[:5])