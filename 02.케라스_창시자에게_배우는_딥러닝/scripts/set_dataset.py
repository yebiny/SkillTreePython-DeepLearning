import shutil, os, glob

def set_dogs_and_cats(org_dir, new_dir):
  org_img_paths = sorted(glob.glob(f'{org_dir}*jpg'))
  print(f'* found {len(org_img_paths)} images in {org_img_paths}')
  os.mkdir(new_dir)
  
  for dir_type in ['train', 'valid', 'test']:
    dir_path = f'{new_dir}/{dir_type}'
    os.mkdir(dir_path)
    print(f'* {dir_path} is made')
    for dir_label in ['cat' , 'dog']:
      dir_path = f'{new_dir}/{dir_type}/{dir_label}'
      os.mkdir(dir_path)
      print(f'  ㄴ{dir_path} is made')

  for img_path in org_img_paths:
    info = img_path.split('/')[-1] # cat.0.jpg
    label, idx, _ = info.split('.')

    if 0<=int(idx)<1000: target_path = f'{new_dir}/train/{label}/{info}'
    elif 1000<=int(idx)<1500: target_path = f'{new_dir}/valid/{label}/{info}'
    else: target_path = f'{new_dir}/test/{label}/{info}'
    #print(label, idx, img_path, target_path)
    shutil.copyfile(img_path, target_path)
  print('* Finished dataset setting')
