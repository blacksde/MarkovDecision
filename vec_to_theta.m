function theta = vec_to_theta(theta_vec)
    theta.nparam = 7;
    theta.VP0 = theta_vec(1);
    theta.VP1 = theta_vec(2);
    theta.VP2 = theta_vec(3);
    theta.FC0 = theta_vec(4);
    theta.FC1 = theta_vec(5);
    theta.EC0 = theta_vec(6);
    theta.EC1 = theta_vec(7);
    theta.pnames = {'VP0','VP1','VP2','FC0','FC1','EC0','EC1'};
end