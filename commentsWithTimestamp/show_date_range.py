from supabase import create_client

SUPABASE_URL = "https://yxfkplhpjgdwgfabrsrf.supabase.co"
SUPABASE_KEY = "sb_publishable_DTSUBBYxEoDpsH5e9HP40g_Q1YiQKeK"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

earliest = (
    supabase.table("YoutubeCommentswTimestamp")
    .select("Comment_Timestamp")
    .order("Comment_Timestamp", desc=False)
    .limit(1)
    .execute()
)

latest = (
    supabase.table("YoutubeCommentswTimestamp")
    .select("Comment_Timestamp")
    .order("Comment_Timestamp", desc=True)
    .limit(1)
    .execute()
)

print("Earliest:", earliest.data[0]["Comment_Timestamp"] if earliest.data else "no data")
print("Latest:  ", latest.data[0]["Comment_Timestamp"] if latest.data else "no data")
