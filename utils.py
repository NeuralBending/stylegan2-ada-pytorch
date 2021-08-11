import torch

def postprocess(img_out):
  return (img_out.permute(0,2,3,1)* 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

def flicker(frames, f = 30):
  frames[::f]*=0
  frames[2::f]*=0
  frames[1::f]=255-frames[1::f]
  return frames

def latent_walk(w,num=20, width=1):
  lin = torch.linspace(0,width,num)[:,None,None]
  # ws=0
  tensor=[]
  for i in range(w.shape[0]-1):
    ws= w[i].repeat((num,1,1))*(width-lin)
    ws+= w[i+1].repeat((num,1,1))*lin
    tensor.append(ws)
  return torch.cat(tensor)

def noiseIt(x,scale=1):
  noise = scale*torch.zeros_like(x).normal_()
  return x+noise

def noiseItT(x,scale=1,n=2):
  app=[]
  random_noise = torch.zeros_like(x).normal_()[1:n]
  lin = torch.linspace(0,1,x.shape[0]//n)[:,None,None,None].cuda()
  for noise in random_noise:
    noise = noise.repeat((x.shape[0]//n,1,1,1))
    
    app+= [noise*lin]
    app+= [noise*(1-lin)]
  noise = torch.cat(app)
  # print (noise.shape, x.shape)
  x+=scale*noise.cuda()
  return x
