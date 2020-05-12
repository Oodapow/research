ipv4str="host    all             all              0.0.0.0/0                       md5"
ipv6str="host    all             all              ::/0                            md5"

if grep -q "$ipv4str" $1 ; then
    echo "Correct IPv4 settings found, doing nothing"
fi

if ! grep -q "$ipv4str" $1 ; then
    echo "No IPv4 settings found, adding one at the end"
    echo "$ipv4str" >> $1
fi

if grep -q "$ipv6str" $1 ; then
    echo "Correct IPv6 settings found, doing nothing"
fi

if ! grep -q "$ipv6str" $1 ; then
    echo "No IPv6 settings found, adding one at the end"
    echo "$ipv6str" >> $1
fi