import math

class Value:
    def __init__(self, data, _children=(), _op='', label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other / self
    
    def __abs__(self):
        out = Value(data= abs(self.data), _children=(self,), _op="abs")

        def _backward():
            self.grad += out.grad * (1.0 if self.data > 0 else -1.0)
        
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        """
        Exponential is no longer limited to scalars. Have to be careful however as the derivative of 'other'
        in respect to 'out', when 'self' is negative, doesn't exist !!
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data= self.data ** other.data, _children=(self, other,), _op=f"**{other.data}")

        def _backward():
            self.grad += out.grad * other.data * self.data ** (other.data -1)
            assert self.data > 0 
            other.grad += out.grad * out.data * math.log(self.data)
        
        out._backward = _backward
        return out
    
    def __rpow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other ** self

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1) / (math.exp(2*x)+1)
        out = Value(t, _children=(self,), _op="tanh")

        def _backward():
            self.grad += out.grad * (1 - out.data ** 2)

        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        t = math.exp(x)
        out = Value(t, _children=(self,), _op="exp")

        def _backward():
            self.grad += out.grad * t

        out._backward = _backward
        return out

    def backward(self):
        
        
        def build_topo(v, topo=[], visited=set()):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo = build_topo(child, topo=topo, visited=visited)
                topo.append(v)  
            return topo
        
        topo = build_topo(self)
        self.grad = 1
        
        for node in reversed(topo):
            node._backward()