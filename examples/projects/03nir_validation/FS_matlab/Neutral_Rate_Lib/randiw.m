function Omega = randiw(Omega0_df, df)
% Omega0_df = Omega(inverse wishart 분포)의 기대값 X 자유도(df)
% inverse wishart 분포를 따르는 Omega의 기대값 = Omega0_df/df;
% df = 자유도, 자유도가 클수록 강한 prior

Omega0_df_inv = invpd(Omega0_df);
Omega_inv = randwishart(Omega0_df_inv, df);
Omega = invpd(Omega_inv);

end