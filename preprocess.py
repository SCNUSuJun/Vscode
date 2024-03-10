import os
import pandas as pd
import cv2
from tqdm import tqdm
from mtcnn import MTCNN



def get_files(path):
    file_info = os.walk(path)
    file_list = []
    for r, d, f in file_info:
        file_list += f
    return file_list


def get_dirs(path):
    file_info = os.walk(path)
    dirs = []
    for d, r, f in file_info:
        dirs.append(d)
    return dirs[1:]



def generate_label_file():
    print('get label....')
    base_dirs = [
        'G:/img/dev/Freeform/save',
        'G:/img/dev/Northwind/save',
        'G:/img/test/Freeform/save',
        'G:/img/test/Northwind/save',
        'G:/img/train/Freeform/save',
        'G:/img/train/Northwind/save'
    ]
    label_base_url = 'G:/img/label/DepressionLabels/'
    labels = []
    for base_dir in base_dirs:
        img_files = get_files(base_dir)
        loader = tqdm(img_files)
        for img_file in loader:
            img_name = os.path.basename(img_file)
            parts = img_name.split('_')
            if len(parts) < 5:
                print(f"Skipping file with unexpected name format: {img_name}")
                continue
            video_id, video_part = parts[:2]
            label_file = f"{video_id}_{video_part}_Depression.csv"
            label_path = os.path.join(label_base_url, label_file)
            if os.path.exists(label_path):
                label = pd.read_csv(label_path, header=None)
                labels.append([img_name, label[0][0]])
                loader.set_description(f'Processing {img_name}')
    
    output_path = 'G:/keep/label.csv'
    pd.DataFrame(labels, columns=['file', 'label']).to_csv(output_path, index=False)
    return labels



def generate_img(path, v_type, img_path):
    videos = get_files(path)
    loader = tqdm(videos)
    for video in loader:
        name = video[:5]
        save_path = img_path + v_type + '/' + name
        os.makedirs(save_path, exist_ok=True)
        cap = cv2.VideoCapture(path + video)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        gap = int(n_frames / 100)
        for i in range(n_frames):
            success, frame = cap.read()
            if success and i % gap == 0:
                cv2.imwrite(save_path + '/{}.jpg'.format(int(i / gap)), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                loader.set_description("data:{} type:{} video:{} frame:{}".format(path.split('/')[2], v_type, name, i))
        cap.release()


'''

def get_face():
    print('get frame faces....')
    detector = MTCNN()
    save_path = ['G:/keep/dev/Freeform/save', 
             'G:/keep/dev/Northwind/save',
             'G:/keep/test/Freeform/save',
             'G:/keep/test/Northwind/save',
             'G:/keep/train/Freeform/save',
             'G:/keep/train/Northwind/save']
    paths = ['G:/img/dev/Freeform/save', 
         'G:/img/dev/Northwind/save',
         'G:/img/test/Freeform/save',
         'G:/img/test/Northwind/save',
         'G:/img/train/Freeform/save',
         'G:/img/train/Northwind/save']
    for index, path in enumerate(paths):
        files = get_files(path)
        loader = tqdm(files)
        for file in loader:
            os.makedirs(save_path[index], exist_ok=True)
            img_path = path + '/' + file
            s_path = save_path[index] + '/' + file
            img = cv2.imread(img_path)
            if img is None:
                print(f"Cannot read image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            info = detector.detect_faces(img)
            if (len(info) > 0):
                x, y, width, height = info[0]['box']
                confidence = info[0]['confidence']
                b, g, r = cv2.split(img)
                img = cv2.merge([r, g, b])
                img = img[y:y + height, x:x + width, :]
                cv2.imwrite(s_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                loader.set_description('confidence:{:4f} img:{}'.format(confidence, img_path))

'''

def get_face(batch_size=100):
    print('get frame faces....')
    detector = MTCNN()
    save_path = ['G:/keep/dev/Freeform/save', 
            'G:/keep/dev/Northwind/save',
            'G:/keep/test/Freeform/save',
            'G:/keep/test/Northwind/save',
            'G:/keep/train/Freeform/save',
            'G:/keep/train/Northwind/save'
            ]
    paths = ['G:/img/dev/Freeform/save', 
        'G:/img/dev/Northwind/save',
        'G:/img/test/Freeform/save',
        'G:/img/test/Northwind/save',
        'G:/img/train/Freeform/save',
        'G:/img/train/Northwind/save'
        ]
    for index, path in enumerate(paths):
        files = get_files(path)
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            loader = tqdm(batch_files)
            for file in loader:
                os.makedirs(save_path[index], exist_ok=True)
                img_path = path + '/' + file
                s_path = save_path[index] + '/' + file
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Cannot read image: {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                info = detector.detect_faces(img)
                if (len(info) > 0):
                    x, y, width, height = info[0]['box']
                    confidence = info[0]['confidence']
                    b, g, r = cv2.split(img)
                    img = cv2.merge([r, g, b])
                    img = img[y:y + height, x:x + width, :]
                    cv2.imwrite(s_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    loader.set_description('confidence:{:4f} img:{}'.format(confidence, img_path))


if __name__ == '__main__':
    # os.makedirs('C:/Users/苏俊/Desktop/大创/AVEC/AVEC2014/keep/img', exist_ok=True)
    # os.makedirs('C:/Users/苏俊/Desktop/大创/AVEC/AVEC2014/keep/processed', exist_ok=True)
    # os.makedirs('C:/Users/苏俊/Desktop/大创/AVEC/AVEC2014/keep/processed/train', exist_ok=True)
    # os.makedirs('C:/Users/苏俊/Desktop/大创/AVEC/AVEC2014/keep/processed/test', exist_ok=True)
    # os.makedirs('C:/Users/苏俊/Desktop/大创/AVEC/AVEC2014/keep/processed/validate', exist_ok=True)
    os.makedirs('G:\\main\\keep\\img', exist_ok=True)
    os.makedirs('G:\\main\\keep\\processed', exist_ok=True)
    os.makedirs('G:\\main\\keep\\processed\\train', exist_ok=True)
    os.makedirs('G:\\main\\keep\\processed\\test', exist_ok=True)
    os.makedirs('G:\\main\\keep\\processed\\validate', exist_ok=True)
    label = generate_label_file() #将运来的label合并为一个csv文件
    #get_img()                     #抽取视频帧，每个视频按间隔抽取100-105帧
    get_face()                    #使用MTCNN提取人脸，并分割图片



