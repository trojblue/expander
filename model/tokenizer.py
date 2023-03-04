import os
import json

class TagTokenizer:
    def __init__(self):
        self.tag_to_id = {}
        self.id_to_tag = {}
        self.vocab_file = 'vocab.json'

        if os.path.exists(self.vocab_file):
            with open(self.vocab_file) as f:
                self.tag_to_id = json.load(f)
                self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        else:
            self.tag_to_id['[PAD]'] = 0
            self.tag_to_id['[UNK]'] = 1
            self.id_to_tag[0] = '[PAD]'
            self.id_to_tag[1] = '[UNK]'

    def tokenize(self, tag_str):
        tags = [tag.strip() for tag in tag_str.split(',')]
        token_ids = [self.tag_to_id.get(tag, self.tag_to_id['[UNK]']) for tag in tags]
        return token_ids

    def expand_tags(self, tags):
        # Replace this with your own tag expansion logic
        expanded_tags = set(tags)
        for tag in tags:
            if tag == 'skirt':
                expanded_tags.add('pleated skirt')
            if tag == 'shoes':
                expanded_tags.add('black footwear')
        return list(expanded_tags)

    def build_vocab(self, tag_list):
        for tag_str in tag_list:
            tags = [tag.strip() for tag in tag_str.split(',')]
            for tag in tags:
                if tag not in self.tag_to_id:
                    tag_id = len(self.tag_to_id)
                    self.tag_to_id[tag] = tag_id
                    self.id_to_tag[tag_id] = tag
        with open(self.vocab_file, 'w') as f:
            json.dump(self.tag_to_id, f)


if __name__ == '__main__':
    # Example usage
    tag_list = [
        'japanese crested ibis (kemono friends), 1girl, skirt, white hair, red pantyhose, head wings, pantyhose, shoes, multicolored hair, solo, long sleeves, pleated skirt, full body, bird tail, black footwear, frilled sleeves, shirt',
        'ayanami rei, 1girl, long hair, solo, plugsuit, very long hair, red eyes, sitting, white bodysuit, breasts, bodysuit, wariza, hair between eyes, neon genesis evangelion, absurdly long hair, evangelion: 3.0+1.0 thrice upon a time'
    ]

    tokenizer = TagTokenizer()
    tokenizer.build_vocab(tag_list)

    tag_ids = tokenizer.tokenize(tag_list[0])
    print(tag_ids)

    expanded_tags = tokenizer.expand_tags(['shoes'])
    print(expanded_tags)