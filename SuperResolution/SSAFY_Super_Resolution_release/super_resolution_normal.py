from PIL import Image


def super_resolution_normal_filter(input,x,mode='BICUBIC'):
    image=Image.open(input);
    print(image)
    if mode=='BICUBIC':
        output=image.resize((int(image.width*x),int(image.height*x)),Image.BICUBIC)
    elif mode=='LANCZOS':
        output=image.resize((int(image.width*x),int(image.height*x)),Image.LANCZOS)
    elif mode=='NEAREST':
        output=image.resize((int(image.width*x),int(image.height*x)),Image.NEAREST)
    elif mode=='BILINEAR':
        output=image.resize((int(image.width*x),int(image.height*x)),Image.BILINEAR)
    else:
        output=image.resize((int(image.width*x),int(image.height*x)),Image.BICUBIC)
        
    return output
    