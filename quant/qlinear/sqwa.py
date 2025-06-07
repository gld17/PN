import torch
from torch import nn
from functools import partial
from ..quant_funcs import *

class WALinear(nn.Module):
	def __init__(self, in_features, out_features, bias=True, act_quant='per_token', a_bit=8, w_bit=8, quantize_output=False, dev='cuda', w_mode='int', a_mode='int', pn_type='offset'):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.a_bit = a_bit
		self.w_bit = w_bit

		self.register_buffer('weight', torch.zeros(self.out_features,
												   self.in_features, dtype=torch.float16, requires_grad=False, device=dev))
		if bias:
			self.register_buffer('bias', torch.zeros(
				(1, self.out_features), dtype=torch.float16, requires_grad=False, device=dev))
		else:
			self.register_buffer('bias', None)

		if act_quant == 'per_token':
			self.act_quant_name = 'per_token'
			self.act_quant = partial(
				quantize_activation_per_token_absmax, n_bits=self.a_bit, mode=a_mode)
		elif act_quant == 'per_tensor':
			self.act_quant_name = 'per_tensor'
			self.act_quant = partial(
				quantize_activation_per_tensor_absmax, n_bits=self.a_bit, mode=a_mode)
		else:
			raise ValueError(f'Invalid act_quant: {act_quant}')

		if quantize_output:
			self.output_quant_name = self.act_quant_name
			self.output_quant = self.act_quant
		else:
			self.output_quant_name = 'None'
			self.output_quant = lambda x: x

	def to(self, *args, **kwargs):
		super(WALinear, self).to(*args, **kwargs)
		self.weight = self.weight.to(*args, **kwargs)
		if self.bias is not None:
			self.bias = self.bias.to(*args, **kwargs)
		return self

	@torch.no_grad()
	def forward(self, x):
		q_x = self.act_quant(x)
		y = torch.functional.F.linear(q_x, self.weight, self.bias)
		q_y = self.output_quant(y)
		return q_y

	@staticmethod
	def from_float(module, weight_quant='per_channel', act_quant='per_token', w_bit=4, a_bit=8, weight_group=128, quantize_output=False, w_mode='int', a_mode='int', model_name=None, module_name=None, pn_type='offset'):
		assert isinstance(module, torch.nn.Linear)
		new_module = WALinear(module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, a_bit=a_bit, w_bit=w_bit, quantize_output=quantize_output, 
							  dev=module.weight.device, w_mode=w_mode, a_mode=a_mode, pn_type=pn_type)
		# Quantize the weight matrices
		if weight_quant == 'per_channel':
			new_module.weight = quantize_weight_per_channel_absmax(module.weight, n_bits=w_bit, mode=w_mode, model_name=model_name, module_name=module_name, pn_type=pn_type)
		elif weight_quant == 'per_tensor':
			new_module.weight = quantize_weight_per_tensor_absmax(module.weight, n_bits=w_bit, mode=w_mode, model_name=model_name, module_name=module_name, pn_type=pn_type)
		elif weight_quant == 'per_group':
			if w_mode=='int':
				new_module.weight = pseudo_quantize_tensor(module.weight, zero_point=False, n_bits=w_bit, q_group_size=weight_group)
			elif w_mode=='fp':
				new_module.weight = pseudo_fp_quantize_tensor(module.weight, zero_point=False, n_bits=w_bit, q_group_size=weight_group)
			elif w_mode=='nf':
				new_module.weight = pseudo_nf_quantize_tensor(module.weight, zero_point=False, n_bits=w_bit, q_group_size=weight_group)
			elif w_mode=='apot':
				new_module.weight = pseudo_apot_quantize_tensor(module.weight, zero_point=False, n_bits=w_bit, q_group_size=weight_group)
			elif w_mode=='pn':
				new_module.weight = pseudo_pn_quantize_tensor(module.weight, zero_point=False, n_bits=w_bit, q_group_size=weight_group, pn_type=pn_type)
			elif w_mode=='dynamic_pn':
				new_module.weight = dynamic_pseudo_pn_quantize_tensor(module.weight, zero_point=False, n_bits=w_bit, q_group_size=weight_group, inplace=True, model_name=model_name, module_name=module_name)
		else:
			raise ValueError(f'Invalid weight_quant: {weight_quant}')
		
		new_module.weight_quant_name = weight_quant
		if module.bias is not None:
			new_module.bias = module.bias
		del module

		return new_module

	def __repr__(self):
		return 'W{}A{}Linear'.format(self.w_bit, self.a_bit)


class WAConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=4, padding=0, groups=None, act_quant='per_token', a_bit=8, w_bit=8, quantize_output=False, dev='cuda', w_mode='int', a_mode='int', pn_type='offset'):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.groups = groups
		
		self.a_bit = a_bit
		self.w_bit = w_bit

		self.register_buffer('weight', torch.zeros(self.in_channels, 1, self.kernel_size[0], dtype=torch.float16, requires_grad=False, device=dev))
		if bias:
			self.register_buffer('bias', torch.zeros(
				(self.out_channels), dtype=torch.float16, requires_grad=False, device=dev))
		else:
			self.register_buffer('bias', None)

		if act_quant == 'per_token':
			self.act_quant_name = 'per_token'
			self.act_quant = partial(
				quantize_activation_per_token_absmax, n_bits=self.a_bit, mode=a_mode)
		elif act_quant == 'per_tensor':
			self.act_quant_name = 'per_tensor'
			self.act_quant = partial(
				quantize_activation_per_tensor_absmax, n_bits=self.a_bit, mode=a_mode)
		else:
			raise ValueError(f'Invalid act_quant: {act_quant}')

		if quantize_output:
			self.output_quant_name = self.act_quant_name
			self.output_quant = self.act_quant
		else:
			self.output_quant_name = 'None'
			self.output_quant = lambda x: x

	def to(self, *args, **kwargs):
		super(WAConv1d, self).to(*args, **kwargs)
		self.weight = self.weight.to(*args, **kwargs)
		if self.bias is not None:
			self.bias = self.bias.to(*args, **kwargs)
		return self
	
	@torch.no_grad()
	def forward(self, x):
		q_x = self.act_quant(x)
		y = torch.functional.F.conv2d(q_x, self.weight, self.bias, self.stride, self.padding, 1, self.groups)
		q_y = self.output_quant(y)
		return q_y

	@staticmethod
	def from_float(module, weight_quant='per_channel', act_quant='per_token', w_bit=4, a_bit=8, weight_group=128, quantize_output=False, w_mode='int', a_mode='int', model_name=None, module_name=None, pn_type='offset'):
		assert isinstance(module, torch.nn.Conv2d)
		new_module = WAConv2d(module.in_channels, module.out_channels, module.kernel_size, module.bias is not None, stride=module.stride[0], padding=module.padding[0], 
							  groups=module.groups, act_quant=act_quant, a_bit=a_bit, w_bit=w_bit, quantize_output=quantize_output, dev=module.weight.device, w_mode=w_mode, 
							  a_mode=a_mode, pn_type=pn_type)

		# Quantize the weight matrices
		if weight_quant == 'per_channel':
			new_module.weight = quantize_weight_per_channel_absmax(module.weight, n_bits=w_bit, mode=w_mode, model_name=model_name, module_name=module_name, pn_type=pn_type)
		elif weight_quant == 'per_tensor':
			new_module.weight = quantize_weight_per_tensor_absmax(module.weight, n_bits=w_bit, mode=w_mode, model_name=model_name, module_name=module_name, pn_type=pn_type)
		else:
			raise ValueError(f'Invalid weight_quant: {weight_quant}')

		new_module.weight_quant_name = weight_quant
		if module.bias is not None:
			new_module.bias = module.bias
		del module
		return new_module

	def __repr__(self):
		return 'W{}A{}Conv2d'.format(self.w_bit, self.a_bit)


class Appro_WAConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=4, padding=0, groups=None, act_quant='per_token', a_bit=8, w_bit=8, quantize_output=False, dev='cuda', w_mode='int', a_mode='int', pn_type='offset'):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.groups = groups
		
		self.a_bit = a_bit
		self.w_bit = w_bit

		self.register_buffer('weight', torch.zeros(self.in_channels, 1, self.kernel_size[0], dtype=torch.float16, requires_grad=False, device=dev))
		if bias:
			self.register_buffer('bias', torch.zeros(
				(self.out_channels), dtype=torch.float16, requires_grad=False, device=dev))
		else:
			self.register_buffer('bias', None)

		if act_quant == 'per_token':
			self.act_quant_name = 'per_token'
			self.act_quant = partial(
				quantize_activation_per_token_absmax, n_bits=self.a_bit, mode=a_mode)
		elif act_quant == 'per_tensor':
			self.act_quant_name = 'per_tensor'
			self.act_quant = partial(
				quantize_activation_per_tensor_absmax, n_bits=self.a_bit, mode=a_mode)
		else:
			raise ValueError(f'Invalid act_quant: {act_quant}')

		if quantize_output:
			self.output_quant_name = self.act_quant_name
			self.output_quant = self.act_quant
		else:
			self.output_quant_name = 'None'
			self.output_quant = lambda x: x

	def to(self, *args, **kwargs):
		super(Appro_WAConv2d, self).to(*args, **kwargs)
		self.weight = self.weight.to(*args, **kwargs)
		if self.bias is not None:
			self.bias = self.bias.to(*args, **kwargs)
		return self
	
	@torch.no_grad()
	def forward(self, x):
		q_x = x
		unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
		y = torch.functional.F.conv2d(q_x, self.weight, self.bias, self.stride, self.padding, 1, self.groups)

		base_m1 = ((q_x.abs()+1e-7)*2**(-torch.floor(torch.log2(q_x.abs()+1e-7))))-1
		base_m2 = ((self.weight.abs()+1e-7)*2**(-torch.floor(torch.log2(self.weight.abs()+1e-7))))-1
		m1 = copy.deepcopy(base_m1)
		m2 = copy.deepcopy(base_m2)
		m1[base_m1>=0.5] = ((1+base_m1)/2)[base_m1>=0.5]
		m1[base_m1<0.5] = (1+base_m1)[base_m1<0.5]
		m2[base_m2>=0.5] = ((1+base_m2)/2)[base_m2>=0.5]
		m2[base_m2<0.5] = (1+base_m2)[base_m2<0.5]

		m2 = m2.reshape(m2.shape[0],-1)[None,:,None,None,:]
		m1 = unfold(m1).reshape(m1.shape[0],m1.shape[1],m2.shape[-1],y.shape[-2],y.shape[-1]).permute(0,1,3,4,2)
		m1[m1==0] = 1
		m1 -= 1
		m2 -= 1
	
		factor = (m1+m2+1)/((m1+1)*(m2+1)).float()
		factor[torch.isinf(factor)]=1
		factor[torch.isnan(factor)]=1
		# import ipdb; ipdb.set_trace

		q_x = unfold(q_x).reshape(m1.shape[0],m1.shape[1],m2.shape[-1],y.shape[-2],y.shape[-1]).permute(0,1,3,4,2)
		weight = self.weight.reshape(self.weight.shape[0],q_x.shape[-1])[None,:,None,None,:]
		q_y = q_x*weight

		q_y = (factor*q_y).sum(-1)
		if self.bias is not None:
			q_y += self.bias

		del y, weight, base_m1, base_m2, factor
		torch.cuda.empty_cache()
		return q_y.float()

	@staticmethod
	def from_float(module, weight_quant='per_channel', act_quant='per_token', w_bit=4, a_bit=8, weight_group=128, quantize_output=False, w_mode='int', a_mode='int', model_name=None, module_name=None):
		assert isinstance(module, torch.nn.Conv2d) and module.groups>1
		new_module = Appro_WAConv2d(module.in_channels, module.out_channels, module.kernel_size, module.bias is not None, stride=module.stride[0], padding=module.padding[0], 
							  groups=module.groups, act_quant=act_quant, a_bit=a_bit, w_bit=w_bit, quantize_output=quantize_output, dev=module.weight.device, w_mode=w_mode, a_mode=a_mode)
		new_module.weight = module.weight
		new_module.weight_quant_name = weight_quant
		if module.bias is not None:
			new_module.bias = module.bias
		del module
		return new_module

	def __repr__(self):
		return 'Appro_W{}A{}Conv2d'.format(self.w_bit, self.a_bit)