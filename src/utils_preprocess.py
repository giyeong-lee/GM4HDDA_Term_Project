from datetime import datetime

def is_weekday(date):
    return datetime(*(list(map(int, date.split('-'))))).weekday() < 5
    
def gFri2eMon(g_fri):
    yyyy, mm, dd = g_fri.split('-')
    dd = int(dd) + 3
    
    if (mm == '03' and dd > 31):
        mm = '04'
        dd = '0' + str(dd-31)
    elif (mm == '04' and dd > 30):
        mm = '05'
        dd = '0' + str(dd-30)
    else:
        dd = '0'*int(dd < 10) + str(dd)
    
    return '-'.join([yyyy, mm, dd])

def is_holiday(date):
    holidays = ['01-01',  # newyear's day
                '05-01',  # worker's day
                '12-25',  # X-mas
                '12-26'   # boxing day
               ]
    
    good_fridays = ['2000-04-21', '2001-04-13', '2002-03-29', '2003-04-18', '2004-04-09',
                    '2005-03-25', '2006-04-14', '2007-04-06', '2008-03-21', '2009-04-10', 
                    '2010-04-02', '2011-04-22', '2012-04-06', '2013-03-29', '2014-04-18', 
                    '2015-04-03', '2016-03-25', '2017-04-14', '2018-03-30', '2019-04-19', 
                    '2020-04-10', '2021-04-02', '2022-04-15'
                   ]
    
    easter_mondays = [gFri2eMon(g_fri) for g_fri in good_fridays]
    
    return (date[-5:] in holidays) or (date in good_fridays + easter_mondays)
