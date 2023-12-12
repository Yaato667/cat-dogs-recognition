# cat-dogs-recognition

Litlle project to learn CNN
to use the code, u need to go to and create an account an take ur API token https://www.kaggle.com/

Import it on collab (kaggle.json)

And in collab launch this commands 

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d salader/dogs-vs-cats

import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

u can lauch the code now ! 
