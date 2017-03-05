
# This represents the world with 0 being a posible state and 1 being a wall
E = [ 0 0 0 0; 0 1 0 0; 0 0 0 0]

# this are the reward for being in any given state
R = [ 0 0 0 1; 0 0 0 -1; 0 0 0 0]

n, m = size(R)
# This are the actions
A = ["UP", "DOWN", "RIGHT", "LEFT"]

# This are all the possible states
S = reshape([(a,b) for a=1:n, b=1:m], 12)

# this is the transition function
function move(s, a, R)
  out = s
  if a == "DOWN"
    out = min(size(R)[1], s[1] + 1), s[2]
  elseif a == "UP"
    out = max(1, s[1] - 1), s[2]
  elseif a == "RIGHT"
    out = s[1], min(size(R)[2], s[2] + 1)
  elseif a == "LEFT"
    out = s[1], max(1, s[2] - 1)
  else
    error("Not a valid action")
  end
  if E[out[1], out[2]] == 1
    out = s
  end
  return out, R[out[1], out[2]]
end



# aux function to see if it is possible to get to a state s2 from s1
function ispossible(s1, s2, a, R)
  res = 0
  if s2 == move(s1, a, R)[1]
    res = 1
  end
  return res
end


# helper function. Return the best action from s given Utility U
function get_max_action(s, A, U, R)
  m = -999
  for a in A
    s2, r = move(s, a,R)
    if m < U[s2[1], s2[2]]
      m = U[s2[1], s2[2]]
    end
  end
  return m
end


# transforms a given U[i,j] to a policy for state (i,j)
function get_policy(s,U, A, R)
  m = -9999
  action = ""
  for a in A
    s2 = move(s,a,R)[1]
    if m < U[s2[1], s2[2]]
      m = U[s2[1], s2[2]]
      action = a
    end
  end
  return action
end


# value iteration function
function value_iteration(U, R, S)
  for s in S
    U[s[1], s[2]] = R[s[1], s[2]] + γ * get_max_action(s,A, U, R)
  end
  return U
end

# Recursive implementation of value iteration
function value_iteration(U, R, S, current, finish)
  if current == finish
    return U
  else
    for s in S
      # Bellmans equation
      U[s[1], s[2]] = R[s[1], s[2]] + γ * get_max_action(s,A, U, R)
      # we can simple use get_max_action since its a Deterministic world
    end
    return value_iteration(U,R,S,current + 1, finish)
  end
end


# constants
γ = 0.1
n_iteration = 100
# starting Utilities are all 0
U = zeros(size(R))

U = value_iteration(U, R, S, 0, n_iteration)
policies = reshape(map(x->get_policy(x,U, A,R), S), (3,4))
