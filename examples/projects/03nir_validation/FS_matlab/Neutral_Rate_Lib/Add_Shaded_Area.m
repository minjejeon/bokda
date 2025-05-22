function Add_Shaded_Area(Xm, ylow, yupp)

   y_ = [ylow ylow yupp yupp];
   
for i = 1:rows(Xm)
   x_ = [Xm(i, 1) Xm(i, 2) Xm(i, 2) Xm(i, 1)];
   p = patch(x_, y_, 'k');
   set(p,'FaceAlpha',0.3);
   set(p,'Linestyle','none');
end

end