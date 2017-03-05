
# This represents the world with 0 being a posible state and 1 being a wall
E = [ 0 0 0 0; 0 1 0 0; 0 0 0 0]

# this are the reward for being in any given state
R = [ 0 0 0 1; 0 0 0 -1; 0 0 0 0]

# This are the actions
A = ["UP", "DOWN", "RIGHT", "LEFT"]

# This are all the possible states
S = reshape([(a,b) for a=1:3, b=1:4], 12)


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




function ispossible(s1, s2, a, R)
  res = 0
  if s2 == move(s1, a, R)[1]
    res = 1
  end
  return res
end


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


function value_iteration(U, R, S)
  for s in S
    U[s[1], s[2]] = R[s[1], s[2]] + γ * get_max_action(s,A, U, R)
  end
  return U
end


γ = 0.1
U = zeros(size(R))


U = value_iteration(U, R, S)
reshape(map(x->get_policy(x,U, A,R), S), (3,4))
