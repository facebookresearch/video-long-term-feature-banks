
def get_model_path(id):
    assert len(id.strip()) == 9
    return ("https://"
        + "dl.fbaipublicfiles.com/video-long-term-feature-banks/{}/model_final.pkl".format(id.strip()))

def get_lfb_model_path(id):
    return ("https://"
        + "dl.fbaipublicfiles.com/video-long-term-feature-banks/{}/lfb_model.pkl".format(id.strip()))


out_lines = []
with open('README.md', 'r') as f:
    for line in f:

        if '[`model`]()' in line or '| method |' in line or '.yaml & epic' in line:
            items = line.split('|')
            items[2], items[3] = items[3], items[2]
            line = '|'.join(items)

        if '[`model`]()' in line:
            id = line.split('|')[-3]
            line = line.replace('[`model`]()', '[`model`]({})'.format(get_model_path(id)))
            line = line.replace('[`lfb model`]()', '[`lfb model`]({})'.format(get_lfb_model_path(id)))

        out_lines.append(line)


with open('README_new.md', 'w') as f:
    for line in out_lines:
        f.write(line)
