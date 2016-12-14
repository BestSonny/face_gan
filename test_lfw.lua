------------------------------------------------------------
--- This code is based on the eyescream code released at
--- https://github.com/facebook/eyescream
------------------------------------------------------------

require 'hdf5'
require 'nngraph'
require 'cudnn'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'sys'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
adversarial = require 'lfw_adverserial'


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs512_lfw64")      subdirectory to save logs
  -n,--network       (default "")          reload pretrained network
  -t,--threads       (default 4)           number of threads
  -b,--batch         (default 100)         number of samples
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  -d,--noiseDim      (default 512)         dimensionality of noise vector
  -w, --window       (default 3)           windsow id of sample image
  --scale            (default 64)          scale of images to train on
]]


if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

print(opt)

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = {3, opt.scale, opt.scale}

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]


print('load test network: ' .. opt.network)
local status, tmp = pcall(torch.load, opt.network)

if not status then
  print('could not find ' .. opt.network .. ' load the default network')
  -- default network
  local filename = paths.concat(opt.save, 'adversarial.net')
  status, tmp = pcall(torch.load, filename)
  if not status then
    return
  end
end



model_D = tmp.D
model_G = tmp.G

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)


local lfwHd5 = hdf5.open('datasets/lfw.hdf5', 'r')
local data = lfwHd5:read('lfw'):all()
data:mul(2):add(-1)
lfwHd5:close()


ntrain = 13000
nval = 233
trainData = data[{{1, ntrain}}]
valData = data[{{ntrain, nval+ntrain}}]

-- this matrix records the current confusion across classes
classes = {'0','1'}
confusion = optim.ConfusionMatrix(classes)

if opt.gpu then
  print('Copy model to gpu')
  model_D:cuda()
  model_G:cuda()
end


-- training loop
while true do
  local noise_inputs = torch.Tensor(opt.batch, opt.noiseDim)
  -- Generate samples
  noise_inputs:normal(0, 1)
  local samples = model_G:forward(noise_inputs)
  samples = nn.HardTanh():forward(samples)
  local to_plot = {}
  for i=1,opt.batch do
    to_plot[#to_plot+1] = samples[i]:float()
  end
  torch.setdefaulttensortype('torch.FloatTensor')
  local formatted = image.toDisplayTensor({input=to_plot, nrow=10})
  formatted:float()
  window = image.display{image = formatted, offscreen = false, win = window}
  sys.sleep(.5)
  --image.save(opt.save .."/lfw_example_v1_"..(epoch or 0)..'.png', formatted)
  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end

end
